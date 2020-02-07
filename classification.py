if __name__ == '__main__':
    # encoder, decoder, x_test = run_simple_autoencoder()
    #
    # # use encoder and decoder to predict
    # encoded_imgs = encoder.predict(x_test)
    # decoded_imgs = decoder.predict(encoded_imgs)
    #
    # valid_x_predictions = encoder.predict(x_test)
    # mse = np.mean(np.power(x_test - valid_x_predictions, 2), axis=1)
    # error_df = pd.DataFrame({'Reconstruction_error': mse,
    #                          'True_class': df_valid['y']})
    # precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    # plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
    # plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
    # plt.title('Precision and recall for different threshold values')
    # plt.xlabel('Threshold')
    # plt.ylabel('Precision/Recall')
    # plt.legend()
    # plt.show()
    exit()