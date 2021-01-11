from my_data_process import get_img_data

tr_path = 'number_images/trainImages/'
te_path = 'number_images/testImages'

tr_data, tr_labels = get_img_data(tr_path)
te_data, te_labels = get_img_data(te_path)