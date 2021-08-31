import argparse
from dataCollector import DataCollector




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a tool for collecting training data, \
    training a neural network model collected data, and running a bot using the output of the neural network.")
    parser.add_argument("-e", "--eta", help="Trains the model with a learning rate set to the value of this argument", type=float)
    parser.add_argument("-d", "--dataset", help="Trains the model on the passed relative path to the dataset.", type=str)
    parser.add_argument("-m", "--model", help="Saves or loads the model in the path under this argument.", type=str)
    parser.add_argument("-c", "--collect", help="Puts the program in data collection mode. (OPT: --dataset dataset_name)")
    parser.add_argument("-t", "--train", help="Puts the program in training mode. (REQ: --dataset dataset_name. OPT: --model model_name, --eta learning_rate)")
    parser.add_argument("-p", "--play", help="Puts the program in playing mode. (REQ: --dataset dataset_name, --model model_name)")
    parser.add_argument("-d", "--debug", help="Sets the program in debug mode which has certain visual changes under play and collect data modes.")
    args = parser.parse_args()
    def validate(illegal, required):
        assert(all(not x for x in illegal))
        assert(all(x for x in required))
    if args.collect:
        illegal = [args.train, args.play, args.model, args.eta]
        required = []
        validate(illegal, required)
        dataCollector = DataCollector()
    elif args.train:
        illegal = [args.play, args.collect]
        required = [args.dataset]
        validate(illegal, required)
    elif args.play:
        illegal = [args.collect, args.train, args.eta]
        required = [args.dataset, args.model]
        validate(illegal, required)
    else:
        raise Exception("Error: No valid argument for program operation.\
            \nUse --collect, --train, or --play to start with an \
            operation.")
