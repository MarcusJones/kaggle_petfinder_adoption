

class RunController():
    def __init__(self, datastruct, model_object):
        self.datastruct = datastruct
        self.model_object = model_object

        self.feature_space = gamete_design_space.DesignSpace(self.datastruct.generate_variables())
        self.model_space = gamete_design_space.DesignSpace(self.model_object.search_grid)

    def get_feature_chromosome(self):
        chromosome = list()
        for var in self.feature_space.basis_set:
            this_var = var.return_random_allele()
            chromosome.append(this_var)
        return chromosome
        logging.info("Generated feature chromosome".format())

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
            if not variable_selection_dict[col]:
                removed_columns.append(col)

        this_df = self.datastruct.df.copy()

        this_df.drop(removed_columns, inplace=True, axis=1)
        return this_df

    def get_model_chromosome(self):
        chromosome = list()
        for var in self.model_space.basis_set:
            this_var = var.return_random_allele()
            chromosome.append(this_var)
        return chromosome

    def get_design_space(self):
        all_vars = self.datastruct.generate_variables() + self.model_object.search_grid
        return gamete_design_space.DesignSpace(all_vars)

this_rc = RunController(ds, model_search)
this_design_space = this_rc.get_design_space()

this_rc.get_feature_chromosome()
this_rc.get_model_chromosome()


this_df = this_rc.selection_dataframe()
print(len(this_df.columns))
# this_feature_selection = this_rc.get_feature_chromosome()
# variable_selection_dict = {var.name:var.value for var in this_feature_selection}
# removed_columns = list()
# for col in this_rc.datastruct.feature_columns:
#     if not variable_selection_dict[col]:
#         removed_columns.append(col)
#
#     # print(col, variable_selection_dict[col])


