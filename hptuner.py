#!/usr/bin/env python
import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([8, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))
HP_LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([0.001, 0.0005, 0.0001]))
HP_OPTIMIZER=hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))

METRIC_ACCURACY = 'RootMeanSquaredError'

log_dir ='\\logs\\fit\\' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT,  HP_OPTIMIZER, HP_LEARNING_RATE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Rmse')],
    )

def train_and_evaluate(model, hparams):
    tensorboard = TensorBoard(log_dir = log_dir, histogram_freq=10, write_graph=True, write_images=False)
    hp_tuning = hp.KerasCallback(log_dir,hparams)
    cbs = [tensorboard,hp_tuning]

    history = model.fit(gen, validation_split=0.15, epochs=p.epochs, batch_size=p.batch_size, callbacks=cbs, verbose=1, shuffle = True)
    mse, rmse = model.evaluate(test_gen)
    return rmse, history

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        model = create_model(hparams)
        accuracy, history = train_and_evaluate(model, hparams)
        #accuracy= tf.reshape(tf.convert_to_tensor(accuracy), []).numpy()
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    return history

def run_tuning():
    session_num = 0
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                    HP_LEARNING_RATE: learning_rate,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                history = run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1

def main(args = sys.argv[1:]):
    args = parseArguments()
    run(args.run_dir_path, args.hparams)


if __name__ == "__main__":
    main()