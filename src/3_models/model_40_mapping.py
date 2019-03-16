class RunController():
    def __init__(self, datastruct, model_object):
        self.datastruct = datastruct
        self.model_object = model_object

        self.feature_space = gamete_design_space.DesignSpace(self.datastruct.generate_variables())
        self.model_space = gamete_design_space.DesignSpace(self.model_object.search_grid)

    def get_feature_chromosome(self):
        chromosome = list()
        for var in self.feature_space.basis_set:
            this_var = var.return_random_allele2()
            chromosome.append(this_var)
        logging.info("Generated feature chromosome".format())
        return chromosome

        # this_ind = self.individual(chromosome=chromosome, )
        # # this_ind = this_ind.init_life_cycle()
        #
        # if flg_verbose:
        #     logging.debug("Creating a {} individual with chromosome {}".format(self.individual, chromosome))
        #     logging.debug("Returned random individual {}".format(this_ind))
        #
        # return this_ind


    def selection_dataframe(self):
        this_feature_selection = self.get_feature_chromosome()
        variable_selection_dict = {var.name: var.value for var in this_feature_selection}
        removed_columns = list()
        for col in self.datastruct.feature_columns:
            # print(col, variable_selection_dict[col], type(variable_selection_dict[col]))
            # if variable_selection_dict[col]: print("YEP")
            if not variable_selection_dict[col]:
                # print("DROP", col)
                removed_columns.append(col)

        this_df = self.datastruct.df.copy()

        this_df.drop(removed_columns, inplace=True, axis=1)
        logging.info("Dropping {} features".format(len(removed_columns)))
        return this_df

    def get_model_chromosome(self):
        chromosome = list()
        for var in self.model_space.basis_set:
            this_var = var.return_random_allele2()
            chromosome.append(this_var)
        logging.info("Generated model chromosome".format())
        return chromosome

    def get_design_space(self):
        all_vars = self.datastruct.generate_variables() + self.model_object.search_grid
        return gamete_design_space.DesignSpace(all_vars)

#%%
this_rc = RunController(ds, model_search)
this_design_space = this_rc.get_design_space()

feature_chromo = this_rc.get_feature_chromosome()
this_var = feature_chromo[0]
this_rc.get_model_chromosome()

#%%
feature_chromo = this_rc.get_feature_chromosome()
feature_chromo_dict = {allele.name:allele.value for allele in feature_chromo}

IND_PATH = Path.cwd() / 'output'
assert IND_PATH.exists()
IND_OUT_FILE = IND_PATH / 'test_feature_chromosome.json'
with IND_OUT_FILE.open('w') as f:
    json.dump(feature_chromo_dict, f)

#%%
model_chromo = this_rc.get_model_chromosome()
model_chromo_dict = {allele.name:allele.value for allele in model_chromo}

IND_PATH = Path.cwd() / 'output'
assert IND_PATH.exists()
IND_OUT_FILE = IND_PATH / 'test_model_chromosome.json'
with IND_OUT_FILE.open('w') as f:
    json.dump(model_chromo_dict, f)

#%%
sub_this_df = this_rc.selection_dataframe()


# this_feature_selection = this_rc.get_feature_chromosome()
# variable_selection_dict = {var.name:var.value for var in this_feature_selection}
# removed_columns = list()
# for col in this_rc.datastruct.feature_columns:
#     if not variable_selection_dict[col]:
#         removed_columns.append(col)
#
#     # print(col, variable_selection_dict[col])


