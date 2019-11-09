import numpy as np
from tensorflow import keras as ks
import data_preproc as dp


def create_branch_of_network(image_shape):
    model = ks.Sequential()
    model.add(ks.layers.Conv2D(64, kernel_size=10, activation='relu', input_shape=image_shape))
    model.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(ks.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(ks.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(ks.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(ks.layers.Conv2D(256, kernel_size=3, activation='relu'))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(4096, activation='sigmoid'))
    return model


def create_siamese_network(image_shape):
    # input images tensors
    image_one = ks.Input(image_shape)
    image_two = ks.Input(image_shape)
    # print(image_shape)
    # CNN
    model = create_branch_of_network(image_shape)

    # Siamese networks predictions
    y_hat_one = model(image_one)
    y_hat_two = model(image_two)

    # Adding layer to compute siaemese networks distance
    distance_computation = ks.layers.Lambda(lambda tensors: ks.backend.abs(tensors[0] - tensors[1]))
    images_distance = distance_computation([y_hat_one, y_hat_two])
    # Dense layer
    y_hat = ks.layers.Dense(1, activation='sigmoid')(images_distance)
    # inputs & outputs
    siamese_net = ks.Model(inputs=[image_one, image_two], outputs=y_hat)

    return siamese_net


def main():
    siamese_in_creator = dp.SiameseDatasetCreator()
    nr_channels, height, width = siamese_in_creator.celeb_loader.dataset[0][0].shape
    model = create_siamese_network((height, width, nr_channels))
    optimizer = ks.optimizers.Adam(lr=0.00006)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    train_siamese_data = siamese_in_creator.generate_verification_input(type_ds='train')
    model.fit_generator(generator=train_siamese_data,
                        steps_per_epoch=len(siamese_in_creator.celeb_loader.train_dataset),
                        use_multiprocessing=True,
                        workers=6)
    test_siamese_data = siamese_in_creator.generate_verification_input(type_ds='test')
    model.evaluate(x=test_siamese_data)

    return


if __name__ == '__main__':
    main()
