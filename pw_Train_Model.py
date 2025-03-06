import os, glob
import yaml
import datetime
import argparse

from ultralytics import YOLO

from pw_Auto_Distill import remove_bad_data

device_dict = {
    'cpu': 'cpu',
    'cuda': 'cuda:0'
    }

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def get_now():
    """

    :return:
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now


def create_training_yaml(yaml_files, output_dir):
    """

    :param yaml_files:
    :param output_dir:
    :return:
    """
    # Initialize variables to store combined data
    combined_data = {'names': [], 'nc': 0, 'train': [], 'val': []}

    try:
        # Iterate through each YAML file
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file)

                # If the class isn't already in the combined list
                if data['names'] not in combined_data['names']:
                    # Combine 'names' field
                    combined_data['names'].extend(data['names'])

                    # Combine 'nc' field
                    combined_data['nc'] += data['nc']

                # Combine 'train' and 'val' paths
                combined_data['train'].append(data['train'])
                combined_data['val'].append(data['val'])

        # Create a new YAML file with the combined data
        output_file_path = f"{output_dir}/training_data.yaml"

        with open(output_file_path, 'w') as output_file:
            yaml.dump(combined_data, output_file)

        # Check that it was written
        if os.path.exists(output_file_path):
            return output_file_path

    except Exception as e:
        raise Exception(f"ERROR: Could not output YAML file!\n{e}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "-data_dir", "-datadir", "-dir", "-root", "-rdir",
                        dest="rdir",
                        type=str,
                        default="/mnt/e/greeen/CabFranc_original/Training_Data/GrapeMapper_detect_0.4_0.5_0.01_JPG",
                        help="The root data directory (Data); OCD")
    parser.add_argument("-o", "-out_dir", "-outdir",
                        dest="outdir",
                        type=str,
                        help="The output data directory")
    parser.add_argument('-dirs', '-num_dirs', '-ndirs',
                        dest='dirs',
                        type=int,
                        default=1,
                        help='Number of directories to process')
    parser.add_argument('-t', '-task',
                        dest='task',
                        type=str,
                        default='segment',
                        choices=['detect', 'segment'],
                        help='Type of task to train')
    parser.add_argument('-dev', '-device',
                        dest='device',
                        type=str,
                        default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('-n', '-num_epochs',
                        dest='num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to train for')
    parser.add_argument('-f', '-file_ext',  
                        dest='file_ext',
                        type=str,
                        default='JPG',
                        help='File extension')
    parser.add_argument('-base', '-base_model',  
                        dest='base_model',
                        type=str,
                        default='yolov8',
                        choices=['yolov8', 'yolov11'],
                        help='Base model to use')
    parser.add_argument('-opt', '-optimizer',  
                        dest='optimizer',
                        type=str,
                        default='Adam',
                        choices=['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'],
                        help='Optimizer to use')

    args = parser.parse_args()

    if not args.outdir:
        args.outdir = os.path.join(os.path.split(args.rdir)[0], "models")

    # Get the root data directory (Data); OCD
    # root = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\Data"
    # root = root.replace("\\", "/")
    # rdir = "B:/RockFinder/images"
    # rdir = "/mnt/e/greeen/CabFranc_original/Training_Data"
    # rdir = "/e/greeen/CabFranc_original/Training_Data"
    print('Input directory: {}'.format(args.rdir))
    
    assert os.path.exists(args.rdir)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    '''
    Dataset Creation
    '''
    yaml_files = []
    if args.dirs > 1:
        # Here we loop though all the datasets in the training_data_dir,
        # get their image / label folders, and the data.yml file.
        # dataset_folders = glob.glob(args.rdir + os.path.sep + '*' + os.path.sep)
        dataset_folders = glob.glob(os.path.join(args.rdir, '*'))
        # dataset_folders = filter(os.path.isdir, os.listdir(training_data_dir))
        [print(d) for d in dataset_folders]

        for dataset_folder in dataset_folders:
            # Get the folder for the dataset
            dataset_folder = f"{dataset_folder}"
            # Remove images and labels from train/valid if they were deleted from rendered
            remove_bad_data(dataset_folder, args.file_ext)
            # Get the YAML file for the dataset
            yaml_file = os.path.join(dataset_folder, "data.yaml")
            assert os.path.exists(yaml_file)
            # Add to the list
            yaml_files.append(yaml_file)
        
        training_yaml = create_training_yaml(yaml_files, os.path.join(args.outdir, "_combined_training_data"))
    else:
        assert os.path.exists(os.path.join(args.rdir, "data.yaml"))
        training_yaml = os.path.join(args.rdir, "data.yaml")

    # Get weights based on task
    if args.task == 'detect':
        if args.base_model == 'yolov8':
            weights = "yolov8n.pt"
        else:
            weights = "yolov11n.pt"
    else:
        if args.base_model == 'yolov8':
            weights = "yolov8n-seg.pt"
        else:
            weights = "yolov11n-seg.pt"

    # Name of the run
    run_name = f"{get_now()}_{weights.split('.')[0]}"

    # Access pre-trained model
    target_model = YOLO(weights)

    # Train model w/ parameters
    results = target_model.train(data=training_yaml,
                                 cache=False,
                                 device=device_dict[args.device],
                                 epochs=args.num_epochs,
                                 patience=int(args.num_epochs * .3),
                                 batch=16,
                                 imgsz=1280,
                                 project=args.outdir,
                                 name=run_name,
                                 optimizer=args.optimizer,
                                 save=True,
                                 save_period=1,
                                 plots=True,
                                 single_cls=True,
                                 )
    