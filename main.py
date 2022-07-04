import click
from experiments.confidence import run_confidence

EXPERIMENT_DICTIONARY = {"confidence" : run_confidence}

@click.command()
@click.option("--experiment", prompt="Which experiment to run")
def main(experiment):
    if experiment in EXPERIMENT_DICTIONARY:
        EXPERIMENT_DICTIONARY[experiment]()

    else:
        print("Currently main doesn't support much else. Try train.py?")

if __name__ == "__main__":
    main()