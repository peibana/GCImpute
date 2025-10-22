import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import torch
from time import time
import numpy as np
import os
import pandas as pd
from parse import parse_args
import sc_handler
import utils
from Graph_AE import Graph_AE_handler




def RUN_MAIN():

    param = dict()
    args = parse_args()
    gpu_id = getattr(args, "gpu", 0)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        param['device'] = torch.device(f"cuda:{gpu_id}")
        torch.cuda.reset_peak_memory_stats(param['device'])  # ✅ 重置峰值显存统计
    else:
        param['device'] = torch.device("cpu")

    logging.info(f"--------> Using device: {param['device']}")
    total_mem = None
    if param['device'].type == "cuda":
        logging.info(f"--------> Using GPU: {torch.cuda.get_device_name(param['device'])}")
        total_mem = torch.cuda.get_device_properties(param['device']).total_memory / 1024 ** 2
        utils.log_gpu_memory(param['device'], note="Start")

    tok_start = time()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    X_sc = sc_handler.sc_load_process(args.sc_data,
                                      csv_to_load=args.csv_to_load,
                                      h5ad_to_load=args.h5ad_to_load,
                                      x10_to_load=args.x10_to_load,
                                      txt_to_load=args.txt_to_load,
                                      transpose=args.transpose,
                                      check_counts=args.check_counts,
                                      filter_min_counts=args.filter_min_counts,
                                      min_gene_counts=args.min_gene_counts,
                                      min_cell_counts=args.min_cell_counts,
                                      logtrans_input=args.logtrans_input,
                                      normalize_amount=args.normalize_amount,
                                      normalize_expression_profile=args.normalize_expression_profile
                                     )

    X_process = X_sc.copy()

    X_imputed = Graph_AE_handler(X_process, args, param)

    logging.info('--------> Outputing results ...')
    df_imputed = pd.DataFrame(data=X_imputed, index=X_sc.obs_names, columns=X_sc.var_names)
    df_imputed.to_csv(os.path.join(args.output_dir, 'imputed.csv'))

    logging.info('--------> Plotting results ...')
    utils.plot(param['graph_AE_loss_list'], ylabel='Graph AE Loss', output_dir=args.output_dir)

    tok_end = time()
    time_used = tok_end - tok_start
    logging.info(f'--------> Total running time (seconds) = {time_used} \n')

    # ✅ 获取全程最大显存
    peak_mem = None
    if param['device'].type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(param['device']) / 1024 ** 2
        logging.info(f"[GPU MEM] Peak allocated memory (total run): {peak_mem:.2f} MB")
    # ✅ 打印每个 epoch 的显存统计
    epoch_peaks = param.get('epoch_peaks')
    if epoch_peaks:
        max_ep = max(epoch_peaks)
        min_ep = min(epoch_peaks)
        avg_ep = sum(epoch_peaks) / len(epoch_peaks)
        logging.info(
            f"[GPU MEM] Epoch peaks -> Max: {max_ep:.2f} MB | "
            f"Min: {min_ep:.2f} MB | Avg: {avg_ep:.2f} MB"
        )
        # ✅ 画显存曲线
        utils.plot(
            epoch_peaks,
            ylabel='GPU Memory (MB)',
            output_dir=args.output_dir,
            suffix='_epoch_peaks'
        )

    # ✅ 保存统计信息
    utils.save_run_stats(
        args.output_dir,
        runtime=time_used,
        peak_mem=peak_mem,
        epoch_peaks=param.get('epoch_peaks'),
        total_mem=total_mem
    )

    logging.info(f'--------> Program Finished! \n')



if __name__ == '__main__':
    RUN_MAIN()






