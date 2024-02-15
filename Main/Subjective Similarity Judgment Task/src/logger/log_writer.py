import pandas as pd

from util.matrix_utils import get_output_matrix_from_embedding


class LogWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def write_embeddings(self, output_embedding, dissim_matrix, matrix_size):  # M_prime, dissim_matrix
        dissim_matrix_from_embedding = get_output_matrix_from_embedding(output_embedding, matrix_size)

        dissim_matrix_df = pd.DataFrame(data=dissim_matrix.astype(float))
        dissim_matrix_df.to_csv(f"{self.output_dir}/dissim_matrix.csv", sep=" ", header=False,
                                float_format="%.6f",
                                index=False)

        output_embedding_df = pd.DataFrame(data=output_embedding.astype(float))
        output_embedding_df.to_csv(f"{self.output_dir}/output_embedding.csv", sep=" ", header=False,
                                   float_format="%.6f",
                                   index=False)

        dm_from_embedding = pd.DataFrame(data=dissim_matrix_from_embedding.astype(float))
        dm_from_embedding.to_csv(f"{self.output_dir}/dissim_matrix_from_embedding.csv", sep=" ", header=False,
                                 float_format="%.6f", index=False)

    def write_run_data(self, run_data_list):
        graph_dataframe = pd.DataFrame(run_data_list)
        graph_dataframe["n_components"] = graph_dataframe["n_components"].astype("category")
        graph_dataframe.to_csv(f"{self.output_dir}/run_data.csv")

    def write_click_data(self, click_data_list):
        click_dataframe = pd.DataFrame(click_data_list)
        click_dataframe.to_csv(f"{self.output_dir}/click_data.csv")

    def write_all(self, click_data_list, run_data_list, output_embedding, dissim_matrix, matrix_size):
        self.write_click_data(click_data_list)
        self.write_run_data(run_data_list)
        self.write_embeddings(output_embedding, dissim_matrix, matrix_size)
