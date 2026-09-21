import torch
from .common import FaissNN, TorchNN
from .patchcore import PatchCore 
from .sampler import IdentitySampler, GreedyCoresetSampler, ApproximateGreedyCoresetSampler 


def get_patchcore(encoder,
                  projectors,
                  device,
                  dtype=torch.float,
                  residual=True,
                  input_shape=(3, 224, 224),
                  pretrain_embed_dimension=1024,
                  target_embed_dimension=1024,
                  patchsize=3,
                  anomaly_scorer_num_nn=1,
                  faiss_on_gpu=True, 
                  faiss_num_workers=8,
                  nn_backend="torch",
                  nn_query_chunk_size=128,
                  coreset_percentage=0.1):
    if not 0 < float(coreset_percentage) < 1:
        raise ValueError("coreset_percentage must be within (0, 1)")
    sampler = get_sampler(name='approx_greedy_coreset', percentage=float(coreset_percentage), device=device)
    
    if nn_backend == "torch":
        nn_method = TorchNN(device=device, query_chunk_size=nn_query_chunk_size)
    elif nn_backend == "faiss":
        nn_method = FaissNN(faiss_on_gpu, faiss_num_workers)
    else:
        raise ValueError(f"Unknown nearest-neighbour backend: {nn_backend}")

    patchcore_instance = PatchCore(device, dtype=dtype)
    patchcore_instance.load(
        encoder=encoder,
        projectors=projectors,
        layers_to_extract_from=None,
        device=device,
        residual=residual,
        input_shape=input_shape,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        featuresampler=sampler,
        anomaly_scorer_num_nn=anomaly_scorer_num_nn,
        nn_method=nn_method,
    )
    
    return patchcore_instance

    
def get_sampler(name, percentage, device):
    if name == "identity":
        return IdentitySampler()
    elif name == "greedy_coreset":
        return GreedyCoresetSampler(percentage, device)
    elif name == "approx_greedy_coreset":
        return ApproximateGreedyCoresetSampler(percentage, device)
