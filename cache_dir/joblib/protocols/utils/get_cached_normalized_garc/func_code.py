# first line: 504
@memory.cache
def get_cached_normalized_garc(gene_name, pos, ref, alt, genome_path, ref_gene_dict, reference, ref_genes):
    return get_normalized_garc(gene_name, pos, ref, alt, reference, ref_genes)
