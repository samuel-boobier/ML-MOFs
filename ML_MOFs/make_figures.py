import cv2
import numpy as np

# # figure 1
# image1 = cv2.imread('Graphs/Figures/Figure 1/BM_CO2_Selectivity.png')
# image2 = cv2.imread('Graphs/Figures/Figure 1/CH4 loading _mol_kg_.png')
# image3 = cv2.imread('Graphs/Figures/Figure 1/CO2 loading _mol_kg_.png')
# image4 = cv2.imread('Graphs/Figures/Figure 1/LOG10 TSN.png')
#
# image5 = cv2.imread('Graphs/Figures/Figure 1/log10_BM_CO2_Selectivity.png')
# image6 = cv2.imread('Graphs/Figures/Figure 1/SC CH4 loading _mol_kg_.png')
# image7 = cv2.imread('Graphs/Figures/Figure 1/SC CO2 loading _mol_kg_.png')
# image8 = cv2.imread('Graphs/Figures/Figure 1/TSN.png')
#
# combined_image = np.hstack((image1, image2, image3, image4))
# combined_image2 = np.hstack((image5, image6, image7, image8))
#
# combined_image3 = np.vstack((combined_image, combined_image2))
#
# cv2.imwrite('Graphs/Figures/Figure 1/Figure_1.png', combined_image3)
#
# # Figure 2
#
# image1 = cv2.imread('Graphs/Figures/Figure 2/RF_TSN.png')
# image2 = cv2.imread('Graphs/Figures/Figure 2/RF_TSN_error_scatter.png')
# image3 = cv2.imread('Graphs/Figures/Figure 2/RF_LOG10 TSN.png')
# image4 = cv2.imread('Graphs/Figures/Figure 2/RF_CO2 loading _mol_kg_.png')
#
# combined_image = np.hstack((image1, image2))
# combined_image2 = np.hstack((image3, image4))
#
# top_right = np.vstack((combined_image, combined_image2))
#
# top_left = cv2.imread('Graphs/Figures/Figure 2/correlation.png')
# top_left = cv2.resize(top_left, (800, 800))
#
# top = np.hstack((top_left, top_right))
#
# image1 = cv2.imread('Graphs/Figures/Figure 2/RF_SC CO2 loading _mol_kg_.png')
# image2 = cv2.imread('Graphs/Figures/Figure 2/SC_CO2_VF.png')
# image3 = cv2.imread('Graphs/Figures/Figure 2/RF_SC CH4 loading _mol_kg_.png')
# image4 = cv2.imread('Graphs/Figures/Figure 2/RF_CH4 loading _mol_kg_.png')
#
# combined_image = np.hstack((image1, image2))
# combined_image2 = np.hstack((image3, image4))
#
# middle_left = np.vstack((combined_image, combined_image2))
#
# middle_right = cv2.imread('Graphs/ML_graphs/Regression/RF_importance.png')
#
# middle = np.hstack((middle_left, middle_right))
#
#
# image1 = cv2.imread('Graphs/ML_graphs/Classification/RF_probs_histogram.png')
# image2 = cv2.imread('Graphs/ML_graphs/Classification/RF_roc_curve.png')
# image3 = cv2.imread('Graphs/ML_graphs/Classification/RF_importance.png')
#
# bottom_upper = np.hstack((image1, image2, image3))
#
# image1 = cv2.imread('Graphs/ML_graphs/Classification/RF_histogram.png')
# table = cv2.imread('Graphs/Figures/Figure 2/table.png')
#
# bottom_lower = np.hstack((image1, table))
#
# bottom = np.vstack((bottom_upper, bottom_lower))
#
# figure = np.vstack((top, middle, bottom))
#
# cv2.imwrite('Graphs/Figures/Figure 2/Figure_2.png', figure)
#
# # Figure 3
#
# image1 = cv2.imread('Graphs/Figures/Figure 3/SC_CO2_VF.png')
# image2 = cv2.imread('Graphs/Figures/Figure 3/RF_SC CO2 loading _mol_kg_.png')
# image3 = cv2.imread('Results/ML_results/Test_set/RF_roc_curve.png')
#
# upper = np.hstack((image1, image2, image3))
#
# image1 = cv2.imread('Results/ML_results/Test_set/RF_histogram.png')
# table = cv2.imread('Graphs/Figures/Figure 3/table.png')
# middle = np.hstack((image1, table))
#
# image1 = cv2.imread('Results/ML_results/Test_set/RF_probs_histogram.png')
# white = np.ones((400, 400, 3), dtype=np.uint8) * 255
# lower = np.hstack((image1, white))
#
# figure = np.vstack((upper, middle, lower))
#
# cv2.imwrite('Graphs/Figures/Figure 3/Figure_3.png', figure)

# cv2.imshow('Combined Image', bottom)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# full figures

# import os
#
# path = "Graphs/Figures/Figure 2"
# for file in os.listdir(path):
#     print(file)
#
# mlr = ["MLR_CH4 loading _mol_kg_.png",
#        "MLR_CO2 loading _mol_kg_.png",
#        "MLR_LOG10 TSN.png",
#        "MLR_SC CH4 loading _mol_kg_.png",
#        "MLR_SC CO2 loading _mol_kg_.png",
#        "MLR_TSN.png"]
#
# ims = [cv2.imread(path + "/" + x) for x in mlr]
#
# top = np.hstack((ims[0], ims[1]))
# middle = np.hstack((ims[2], ims[3]))
# bottom = np.hstack((ims[4], ims[5]))
#
# figure = np.vstack((top, middle, bottom))
#
# cv2.imwrite('Graphs/Figures/Figure 2/mlr_full.png', figure)
#
#
# rf = ["RF_CH4 loading _mol_kg_.png",
#        "RF_CO2 loading _mol_kg_.png",
#        "RF_LOG10 TSN.png",
#        "RF_SC CH4 loading _mol_kg_.png",
#        "RF_SC CO2 loading _mol_kg_.png",
#        "RF_TSN.png"]
#
# ims = [cv2.imread(path + "/" + x) for x in rf]
#
# top = np.hstack((ims[0], ims[1]))
# middle = np.hstack((ims[2], ims[3]))
# bottom = np.hstack((ims[4], ims[5]))
#
# figure = np.vstack((top, middle, bottom))
#
# cv2.imwrite('Graphs/Figures/Figure 2/rf_full.png', figure)
#
#
# svm = ["SVM_CH4 loading _mol_kg_.png",
#        "SVM_CO2 loading _mol_kg_.png",
#        "SVM_LOG10 TSN.png",
#        "SVM_SC CH4 loading _mol_kg_.png",
#        "SVM_SC CO2 loading _mol_kg_.png",
#        "SVM_TSN.png"]
#
# ims = [cv2.imread(path + "/" + x) for x in svm]
#
# top = np.hstack((ims[0], ims[1]))
# middle = np.hstack((ims[2], ims[3]))
# bottom = np.hstack((ims[4], ims[5]))
#
# figure = np.vstack((top, middle, bottom))
#
# cv2.imwrite('Graphs/Figures/Figure 2/svm_full.png', figure)

# full error plots

# import os
#
# path = "Graphs/Figures/Scatter Error"
# for file in os.listdir(path):
#     print(file)
#
# mlr = ["MLR_CH4 loading _mol_kg__error_scatter.png",
#        "MLR_CO2 loading _mol_kg__error_scatter.png",
#        "MLR_LOG10 TSN_error_scatter.png",
#        "MLR_SC CH4 loading _mol_kg__error_scatter.png",
#        "MLR_SC CO2 loading _mol_kg__error_scatter.png",
#        "MLR_TSN_error_scatter.png"]
#
# ims = [cv2.imread(path + "/" + x) for x in mlr]
#
# top = np.hstack((ims[0], ims[1]))
# middle = np.hstack((ims[2], ims[3]))
# bottom = np.hstack((ims[4], ims[5]))
#
# figure = np.vstack((top, middle, bottom))
#
# cv2.imwrite('Graphs/Figures/Scatter Error/mlr_full_errors.png', figure)
#
#
# rf = ["RF_CH4 loading _mol_kg__error_scatter.png",
#        "RF_CO2 loading _mol_kg__error_scatter.png",
#        "RF_LOG10 TSN_error_scatter.png",
#        "RF_SC CH4 loading _mol_kg__error_scatter.png",
#        "RF_SC CO2 loading _mol_kg__error_scatter.png",
#        "RF_TSN_error_scatter.png"]
#
# ims = [cv2.imread(path + "/" + x) for x in rf]
#
# top = np.hstack((ims[0], ims[1]))
# middle = np.hstack((ims[2], ims[3]))
# bottom = np.hstack((ims[4], ims[5]))
#
# figure = np.vstack((top, middle, bottom))
#
# cv2.imwrite('Graphs/Figures/Scatter Error/rf_full_errors.png', figure)
#
#
# svm = ["SVM_CH4 loading _mol_kg__error_scatter.png",
#        "SVM_CO2 loading _mol_kg__error_scatter.png",
#        "SVM_LOG10 TSN_error_scatter.png",
#        "SVM_SC CH4 loading _mol_kg__error_scatter.png",
#        "SVM_SC CO2 loading _mol_kg__error_scatter.png",
#        "SVM_TSN_error_scatter.png"]
#
# ims = [cv2.imread(path + "/" + x) for x in svm]
#
# top = np.hstack((ims[0], ims[1]))
# middle = np.hstack((ims[2], ims[3]))
# bottom = np.hstack((ims[4], ims[5]))
#
# figure = np.vstack((top, middle, bottom))
#
# cv2.imwrite('Graphs/Figures/Scatter Error/svm_full_errors.png', figure)

# roc curves in full
image1 = cv2.imread('Graphs/ML_graphs/Classification/KNN_roc_curve.png')
image2 = cv2.imread('Graphs/ML_graphs/Classification/RF_roc_curve.png')
image3 = cv2.imread('Graphs/ML_graphs/Classification/SVM_roc_curve.png')

figure = np.hstack((image1, image2, image3))

cv2.imwrite('Graphs/Figures/Additional Figures/roc.png', figure)

# classification in full
image1 = cv2.imread('Graphs/ML_graphs/Classification/KNN_histogram.png')
image2 = cv2.imread('Graphs/ML_graphs/Classification/RF_histogram.png')
image3 = cv2.imread('Graphs/ML_graphs/Classification/SVM_histogram.png')
image4 = cv2.imread('Graphs/ML_graphs/Classification/KNN_probs_histogram.png')
image5 = cv2.imread('Graphs/ML_graphs/Classification/RF_probs_histogram.png')
image6 = cv2.imread('Graphs/ML_graphs/Classification/SVM_probs_histogram.png')


figure = np.vstack((image1, image2, image3, image4, image5, image6))

cv2.imwrite('Graphs/Figures/Additional Figures/classification_hist.png', figure)

