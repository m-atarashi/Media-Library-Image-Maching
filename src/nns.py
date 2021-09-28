import faiss

def faiss_knnmatch(resource, query_descriptor, train_descriptor, is_binary=True, nlist=10):
    dimension = query_descriptor.shape[1]

    if is_binary:
        index_flat = faiss.GpuIndexBinaryFlat(resource, 8*dimension)
    else:
        ivfflat_config = faiss.GpuIndexIVFFlatConfig()
        index_flat     = faiss.GpuIndexIVFFlat(resource, dimension, nlist, faiss.METRIC_L2, ivfflat_config)
        index_flat.setNumProbes(2)

    index_flat.add(train_descriptor)
    D, I = index_flat.search(query_descriptor, k=2)
    return D, I