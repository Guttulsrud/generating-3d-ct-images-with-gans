import os
import tempfile

from monai.apps import download_and_extract
from monai.apps.auto3dseg import (
    DataAnalyzer,
    BundleGen,
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
    export_bundle_algo_history,
    import_bundle_algo_history,
)

work_dir = './256_524F_testing_ensemble'

bundle_generator = BundleGen(
    algo_path=work_dir,
    data_stats_filename='datastats.yaml',
    data_src_cfg_name='input.yaml',
)

#bundle_generator.generate(work_dir, num_fold=5)

print("algo path: ", os.path.abspath(work_dir))
print("data_stats file: ", os.path.abspath('datastats.yaml'))
print("task input file: ", os.path.abspath('input.yaml'))

#history = import_bundle_algo_history('256_524F_testing_ensemble', only_trained=True)
#input = "./input.yaml"
#builder = AlgoEnsembleBuilder(history, input)
#builder.set_ensemble_method(AlgoEnsembleBestN(n_best=5))
#ensembler = builder.get_ensemble()

#preds = ensembler()
history = bundle_generator.get_history()
export_bundle_algo_history(history)
builder = AlgoEnsembleBuilder(history, 'input.yaml')
builder.set_ensemble_method(AlgoEnsembleBestN(n_best=5))
ensembler = builder.get_ensemble()
preds = ensembler()