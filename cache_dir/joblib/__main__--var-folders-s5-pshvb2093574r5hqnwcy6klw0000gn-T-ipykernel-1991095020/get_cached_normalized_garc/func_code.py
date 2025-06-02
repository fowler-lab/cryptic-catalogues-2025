# first line: 1
@memory.cache
def get_cached_normalized_garc(gene_name, pos, ref, alt, reference, ref_genes):
    return utils.get_normalized_garc(gene_name, pos, ref, alt, reference, ref_genes)
