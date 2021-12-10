import pandas
import gen_plots as myplt

results = pandas.read_csv(r'./experiment/exp2_data.csv')
myplt.plot_exp_results(results=results)