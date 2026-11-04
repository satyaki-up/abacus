import torch
import torch.distributed as dist


def _has_parameters(module: torch.nn.Module) -> bool:
    return next(module.parameters(), None) is not None


class _Partition:
    """A flat parameter range, padded so every rank owns the same number of elements."""

    def __init__(self, params: list[torch.nn.Parameter], rank: int, world_size: int):
        if not params:
            raise ValueError("parameter partition is empty")
        self.params = params
        self.shapes = [tuple(param.shape) for param in params]
        self.numels = [param.numel() for param in params]
        self.offsets: list[int] = []
        offset = 0
        for numel in self.numels:
            self.offsets.append(offset)
            offset += numel
        self.total = offset
        # reduce_scatter needs equal chunks. The tail of the last chunk is padding.
        self.shard = (self.total + world_size - 1) // world_size
        self.padded = self.shard * world_size
        start = rank * self.shard
        end = start + self.shard
        self.owns: list[tuple[int, int]] = []
        self.dests: list[int] = []
        for param_offset, numel in zip(self.offsets, self.numels):
            begin = max(0, start - param_offset)
            stop = min(numel, end - param_offset)
            if begin >= stop:
                self.owns.append((0, 0))
                self.dests.append(0)
            else:
                self.owns.append((begin, stop))
                self.dests.append(max(param_offset, start) - start)
        # Stage 3 keeps these. Empty until the full parameters have been broadcast.
        self.shards: list[torch.Tensor] = []
        self.gathered = False
        self.pending: dict[int, torch.Tensor] = {}
        self.reduced = False


class DataParallel:
    """Replica of one model per rank. Gradients are the mean over every rank's tokens.

    Stage 0 replicates parameters, gradients, and Adam moments.
    Stage 1 shards the moments. Stage 2 also shards the gradients.
    Stage 3 also shards the parameters, and gathers one module at a time.
    """

    def __init__(self, model: torch.nn.Module, stage: int = 0):
        if not dist.is_initialized():
            raise RuntimeError("Call init_process_group before DataParallel")
        if stage not in (0, 1, 2, 3):
            raise ValueError(f"zero stage must be 0, 1, 2, or 3, got {stage}")
        self.model = model
        self.stage = stage
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.partitions: list[_Partition] = []
        # id(param) -> (begin, end) into the flattened full parameter. None on stage 0.
        self.owned: dict[int, tuple[int, int]] | None = None
        # Stage 2 and 3 store the owned gradient here. param.grad cannot shrink.
        self.grad_shards: dict[int, torch.Tensor] | None = None
        self._token_count = 0


    def set_token_count(self, token_count: int) -> None:
        # Stage 3 multiplies gradients by this inside the backward hooks.
        self._token_count = token_count
        for part in self.partitions:
            part.pending.clear()
            part.reduced = False


    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        # performed once at the start to distribute the model weights to all ranks
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        for buf in self.model.buffers():
            dist.broadcast(buf, src=0)
        if self.stage == 0:
            return
        self.partitions = []
        if self.stage == 3:
            self._install_stage3()
        else:
            self.partitions.append(
                _Partition(list(self.model.parameters()), self.rank, self.world_size)
            )
        self.owned = {}
        for part in self.partitions:
            for param, (begin, stop) in zip(part.params, part.owns):
                if begin != stop:
                    self.owned[id(param)] = (begin, stop)
        self.grad_shards = {} if self.stage >= 2 else None


    @torch.no_grad()
    def synchronize(self, loss: torch.Tensor, token_count: int) -> torch.Tensor:
        if self.stage == 0:
            return self._synchronize_replicated(loss, token_count)
        mean, total = self._mean_loss(loss, token_count)
        if self.stage == 1:
            self._all_reduce_grads(token_count, total)
        elif self.stage == 2:
            self._reduce_scatter_grads(token_count, total)
        else:
            self._scale_sharded_grads(total)
        return mean


    @torch.no_grad()
    def gather_parameters(self) -> None:
        # Stages 1 and 2 wrote one slice per rank. The next forward needs every slice.
        if self.stage not in (1, 2):
            return
        part = self.partitions[0]
        flat = self._all_gather_unpadded(part, from_shards=False)
        for param, offset, numel in zip(part.params, part.offsets, part.numels):
            param.data.reshape(-1).copy_(flat[offset : offset + numel])


    @torch.no_grad()
    def full_parameters(self) -> list[torch.Tensor]:
        if self.stage < 3:
            return [param.detach().clone() for param in self.model.parameters()]
        by_id: dict[int, torch.Tensor] = {}
        for part in self.partitions:
            flat = self._all_gather_unpadded(part, from_shards=True)
            for param, offset, numel, shape in zip(part.params, part.offsets, part.numels, part.shapes):
                by_id[id(param)] = flat[offset : offset + numel].view(shape).clone()
        return [by_id[id(param)] for param in self.model.parameters()]


    def _synchronize_replicated(self, loss: torch.Tensor, token_count: int) -> torch.Tensor:
        grads = [param.grad for param in self.model.parameters() if param.grad is not None]
        if not grads:
            raise RuntimeError("backward produced no gradients")
        # One vector, so the reduction is a single collective instead of one per parameter.
        flat = torch.cat([grad.reshape(-1) for grad in grads])
        loss_sum = loss.detach().reshape(1) * token_count
        count = torch.tensor([token_count], dtype=flat.dtype, device=flat.device)
        # Gradient sums, then the summed loss, then the token count
        payload = torch.cat([flat * token_count, loss_sum, count])
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        total_tokens = payload[-1]
        if total_tokens.item() == 0:
            raise RuntimeError("no tokens contributed to the loss")
        reduced = payload[:-1] / total_tokens
        offset = 0
        for grad in grads:
            numel = grad.numel()
            grad.copy_(reduced[offset : offset + numel].view_as(grad))
            offset += numel
        return reduced[-1]


    def _mean_loss(self, loss: torch.Tensor, token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        packed = torch.stack((
            loss.detach().reshape(()) * token_count,
            torch.tensor(float(token_count), dtype=loss.dtype, device=loss.device),
        ))
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        total = packed[1]
        if total.item() == 0:
            raise RuntimeError("no tokens contributed to the loss")
        return packed[0] / total, total


    def _all_reduce_grads(self, token_count: int, total: torch.Tensor) -> None:
        part = self.partitions[0]
        flat = self._packed_grads(part, token_count)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat = flat[: part.total] / total
        for param, offset, numel in zip(part.params, part.offsets, part.numels):
            param.grad.copy_(flat[offset : offset + numel].view_as(param.grad))


    def _reduce_scatter_grads(self, token_count: int, total: torch.Tensor) -> None:
        part = self.partitions[0]
        self._scatter_owned_grads(part, self._packed_grads(part, token_count), total)
        for param in part.params:
            param.grad = None


    def _scale_sharded_grads(self, total: torch.Tensor) -> None:
        for part in self.partitions:
            if not part.reduced:
                raise RuntimeError("backward produced no gradient for a parameter")
            part.reduced = False
        for grad in self.grad_shards.values():
            grad.div_(total)


    def _packed_grads(self, part: _Partition, token_count: int) -> torch.Tensor:
        flat = torch.zeros(
            part.padded, dtype=part.params[0].dtype, device=part.params[0].device
        )
        for param, offset, numel in zip(part.params, part.offsets, part.numels):
            if param.grad is None:
                raise RuntimeError("backward produced no gradient for a parameter")
            flat[offset : offset + numel].copy_(param.grad.reshape(-1))
        flat.mul_(token_count)
        return flat


    def _scatter_owned_grads(
        self,
        part: _Partition,
        flat: torch.Tensor,
        total: torch.Tensor | None,
    ) -> None:
        local = flat.new_empty(part.shard)
        dist.reduce_scatter_single(local, flat, op=dist.ReduceOp.SUM)
        if total is not None:
            local.div_(total)
        for param, (begin, stop), dest in zip(part.params, part.owns, part.dests):
            if begin == stop:
                continue
            self.grad_shards[id(param)] = local[dest : dest + (stop - begin)].clone()


    def _install_stage3(self) -> None:
        modules = self._unit_modules()
        covered: set[int] = set()
        for module in modules:
            params = list(module.parameters())
            if not params:
                continue
            part = _Partition(params, self.rank, self.world_size)
            self._materialize(part)
            self.partitions.append(part)
            covered.update(id(param) for param in params)
            self._hook_module(module, part)
        root_params = [param for param in self.model.parameters() if id(param) not in covered]
        if not root_params:
            return
        part = _Partition(root_params, self.rank, self.world_size)
        self._materialize(part)
        self.partitions.append(part)
        self._hook_root_parameters(part)


    def _unit_modules(self) -> list[torch.nn.Module]:
        # One collective per block. Splitting a block further would gather one matrix at a time.
        blocks = getattr(self.model, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList):
            return [self.model]
        modules: list[torch.nn.Module] = []
        embedding = getattr(self.model, "embedding", None)
        if embedding is not None and _has_parameters(embedding):
            modules.append(embedding)
        modules.extend(block for block in blocks if _has_parameters(block))
        norm = getattr(self.model, "norm", None)
        if norm is not None and _has_parameters(norm):
            modules.append(norm)
        return modules


    def _materialize(self, part: _Partition) -> None:
        for param, (begin, stop) in zip(part.params, part.owns):
            if begin == stop:
                shard = param.data.new_empty(0)
            else:
                shard = param.data.reshape(-1)[begin:stop].detach().clone()
            part.shards.append(shard)
        for param, shard in zip(part.params, part.shards):
            param.data = shard
        part.gathered = False


    def _hook_module(self, module: torch.nn.Module, part: _Partition) -> None:
        module.register_forward_pre_hook(lambda _module, _args, part=part: self._gather(part))
        module.register_forward_hook(lambda _module, _args, _out, part=part: self._reshard(part))
        module.register_full_backward_pre_hook(
            lambda _module, _grad_output, part=part: self._gather(part)
        )
        for param in part.params:
            param.register_post_accumulate_grad_hook(
                lambda p, part=part: self._on_param_grad(p, part)
            )


    def _hook_root_parameters(self, part: _Partition) -> None:
        # The logits matrix is applied after the final norm, so it is gathered on the way out.
        norm = getattr(self.model, "norm", None)
        if isinstance(norm, torch.nn.Module):
            norm.register_forward_hook(lambda _module, _args, _out, part=part: self._gather(part))
        else:
            self.model.register_forward_pre_hook(lambda _module, _args, part=part: self._gather(part))
        self.model.register_forward_hook(lambda _module, _args, _out, part=part: self._reshard(part))
        self.model.register_full_backward_pre_hook(
            lambda _module, _grad_output, part=part: self._gather(part)
        )
        for param in part.params:
            param.register_post_accumulate_grad_hook(
                lambda p, part=part: self._on_param_grad(p, part)
            )


    @torch.no_grad()
    def _gather(self, part: _Partition) -> None:
        if part.gathered:
            return
        flat = self._all_gather_unpadded(part, from_shards=True)
        for param, offset, numel, shape in zip(part.params, part.offsets, part.numels, part.shapes):
            param.data = flat[offset : offset + numel].view(shape).clone()
        part.gathered = True


    @torch.no_grad()
    def _reshard(self, part: _Partition) -> None:
        if not part.gathered:
            return
        # The autograd graph holds the parameter. Swapping .data does not bump its version,
        # and the backward pre-hook puts the full parameter back before it is read.
        for param, shard in zip(part.params, part.shards):
            param.data = shard
        part.gathered = False


    @torch.no_grad()
    def _on_param_grad(self, param: torch.nn.Parameter, part: _Partition) -> None:
        if param.grad is None:
            raise RuntimeError("backward produced no gradient for a parameter")
        if id(param) in part.pending:
            raise RuntimeError("gradient accumulated twice for one parameter")
        part.pending[id(param)] = param.grad.detach().clone()
        param.grad = None
        if len(part.pending) != len(part.params):
            return
        flat = torch.zeros(part.padded, dtype=param.dtype, device=param.device)
        for owned, offset, numel in zip(part.params, part.offsets, part.numels):
            flat[offset : offset + numel].copy_(part.pending[id(owned)].reshape(-1))
        flat.mul_(self._token_count)
        self._scatter_owned_grads(part, flat, None)
        part.pending.clear()
        self._reshard(part)
        part.reduced = True


    def _all_gather_unpadded(self, part: _Partition, from_shards: bool) -> torch.Tensor:
        local = torch.zeros(part.shard, dtype=part.params[0].dtype, device=part.params[0].device)
        for index, (param, (begin, stop), dest) in enumerate(zip(part.params, part.owns, part.dests)):
            if begin == stop:
                continue
            if from_shards:
                src = part.shards[index]
            else:
                src = param.data.reshape(-1)[begin:stop]
            local[dest : dest + (stop - begin)].copy_(src)
        gathered = local.new_empty(part.padded)
        dist.all_gather_single(gathered, local)
        return gathered[: part.total]
