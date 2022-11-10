from src.plots import plot_figures, plot_table
from src.evaluation import concept_based, test_based

import time

def main():
    total_mark = 100
    tasks = {
             # 1 : 'search',
             2 : 'unique_day',
             # 3 : 'remove_extras',
             # 4 : 'sort_age',
             # 5 : 'top_k'
    }
    # concept_based.run_concept_based(tasks)
    # test_based.run_test_based(tasks)
    plot_table.plot_table_3([1,2,3,4,5], [0,25,50,75])
    plot_tasks = [1,2,3,4,5]
    # plot_table.plot_time_taken(plot_tasks, 0)
    # plot_figures.plot_figure4(plot_tasks)
    plot_table.plot_table_2(plot_tasks, 0)
    plot_table.plot_table_2(plot_tasks, 20)
    plot_table.plot_table_2(plot_tasks, 40)
    plot_table.plot_table_2(plot_tasks, 60)
    plot_table.plot_table_2(plot_tasks, 80)
if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print((stop-start)/60)
