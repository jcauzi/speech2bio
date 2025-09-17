import sys
from plumbum import local, FG
from plumbum.commands.processes import ProcessExecutionError
import argparse

python = local['python']
local['mkdir']['-p', 'logs']()

parser = argparse.ArgumentParser(description='Specify arguments for model parameters')
parser.add_argument('-f', '--fine_tune', action='store_true', help='Wether to fine_tune the model or not', required=False)
parser.add_argument('-t', '--task', nargs='?', const=None, help="Specify the task (classification or detection) (True if provided with a value, False if not provided or without value)"
)
parser.add_argument('-d', '--dataset', type=str, help='The name of the dataset for the specified task', required=False)
parser.add_argument('-m', '--model_type', type=str, help='The name of the model type ("HuBERT", "AVES_bio", HuBERT_base", "WavLM", XEUS)', required=True)
parser.add_argument('-b', '--batch_size', type=str, help='The batch size', required=True)
parser.add_argument('-l', '--layer_num', type=int, help='The layer used for feature extraction', required=True)
parser.add_argument('-lr', '--lrs', type=str, help='list of learning rates to test (check log files if reruning after crash)', required=True)
parser.add_argument('-p', '--probe', type=str, help='probe type : linear, rand_linear', required=True)

args = parser.parse_args()

# MODELS = [
#     ('lr', 'lr', '{"C": [0.1, 1.0, 10.0]}'),
#     ('svm', 'svm', '{"C": [0.1, 1.0, 10.0]}'),
#     ('decisiontree', 'decisiontree', '{"max_depth": [None, 5, 10, 20, 30]}'),
#     ('gbdt', 'gbdt', '{"n_estimators": [10, 50, 100, 200]}'),
#     ('xgboost', 'xgboost', '{"n_estimators": [10, 50, 100, 200]}'),
#     ('resnet18', 'resnet18', ''),
#     ('resnet18-pretrained', 'resnet18-pretrained', ''),
#     ('resnet50', 'resnet50', ''),
#     ('resnet50-pretrained', 'resnet50-pretrained', ''),
#     ('resnet152', 'resnet152', ''),
#     ('resnet152-pretrained', 'resnet152-pretrained', ''),
#     ('vggish', 'vggish', ''),
# ]

MODELS = [
    (args.model_type, args.model_type, '')
]

# TASKS = [
#     ('classification', 'watkins'),
#     ('classification', 'bats'),
#     ('classification', 'dogs'),
#     ('classification', 'cbi'),
#     ('classification', 'humbugdb'),
#     ('detection', 'dcase'),
#     ('detection', 'enabirds'),
#     ('detection', 'hiceas'),
#     ('detection', 'hainan-gibbons'),
#     ('detection', 'rfcx'),
#     ('classification', 'esc50'),
#     ('classification', 'speech-commands'),
# ]


#working batch sizes : 
#humbugdb 12
#dcase 12
#watkins 12
#dogs 12 ?????
#enabirds 12 ?????
#hiceas 12 ?????
#hainan-gibbons 12 ??????
#rfcx 12 ?????
#esc50 12 ?????
#speech-commands 12 ?????

#cuda issues : 
#bats
#cbi
#rfcx


if args.task :
    TASKS = [
        (args.task, args.dataset)
    ]
    print("specified")
else : 
    TASKS = [
        ('classification', 'watkins')
    ]
    print("Not specified")

for model_name, model_type, model_params in MODELS:
    for task, dataset in TASKS:

        #print("SAVING AS DOWN_PITCHED !!!!!!!!!!!!!!!!!!!!!!!!! (remove mention in run_benchmark and change wav path name for bats and cbi if unwanted !!!!!!!!!!!!)")

        print(f'Running {dataset}-{model_name} (layer {args.layer_num} - fine_tune: {args.fine_tune})', file=sys.stderr)
        log_path = f'logs/{dataset}-{model_name}_l{args.layer_num}_ft' if args.fine_tune else f'logs/{args.probe}/{dataset}-{model_name}_l{args.layer_num}_fz_rerun_s'
        #log_path = f'logs/{dataset}-{model_name}_l{args.layer_num}_ft' if args.fine_tune else f'logs/{args.probe}/{dataset}-{model_name}_l{args.layer_num}_fz_down_pitched'
        try:
            if model_type in ['lr', 'svm', 'decisiontree', 'gbdt', 'xgboost']:
                python[
                    'scripts/evaluate.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--params', model_params,
                    '--log-path', log_path,
                    '--num-workers', '4'] & FG
            else:
                python[
                    'scripts/evaluate_new.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--batch-size', args.batch_size,
                    '--epochs', '100',
                    '--lrs', args.lrs,
                    '--log-path', log_path,
                    '--num-workers', '1',
                    '--fine_tune', args.fine_tune,
                    '--layer_num', args.layer_num,
                    '--probe', args.probe] & FG
        except ProcessExecutionError as e:
            print(e)


#lrs : '[1e-5, 5e-5, 1e-4]'