

ds = ds
model_search = model_search
ds.dtypes()
ds.search_grid
vset_feature = gamete.design_space.VariableList("feature set", list())
vset_model = gamete.design_space.VariableList("model set", list())

feature_names = [vdef['name'] for vdef in ds.search_grid]
hyper_param_names = [vdef['name'] for vdef in model_search.search_grid]
assert not list(set(feature_names) & set(hyper_param_names))

for vdef in ds.search_grid:
    vset_feature.append(gamete.design_space.Variable(**vdef))

for vdef in model_search.search_grid:
    vset_model.append(gamete.design_space.Variable(**vdef))


this_ds = gamete.design_space.DesignSpace([vset_feature, vset_model])
print(this_ds)

this_ds.print_design_space()

this_chromo = this_ds.gen_chromosome()

this_ind = gamete.evolution_space.Genome(this_chromo)
print(this_ind)

