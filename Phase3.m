[coeff_1, scores_1, latent_1] = pca(spoon_master);


tree = fitctree(scores_1((1:711), : ),Y_master((1:711), : ));
label_tree = predict(tree,scores_1((1107:1185), : ));

C_tree = confusionmat(Y_master((1107:1185), : )',label_tree);

master_recall_tree = C_tree(2,2)/sum(C_tree(2,:));
master_precision_tree = C_tree(2,2)/sum(C_tree(:,2));
master_Fscore_tree = (2*master_precision_tree*master_recall_tree)/(master_precision_tree+master_recall_tree);

svm_11 = fitcsvm(scores_1((1:711), : ),Y_master((1:711), : ));
label_svm_11 = predict(svm_11,scores_1((1107:1185), : ));

C_svm = confusionmat(Y_master((1107:1185), : )',label_svm_11);

master_recall_svm = C_svm(2,2)/sum(C_svm(2,:));
master_precision_svm = C_svm(2,2)/sum(C_svm(:,2));
master_Fscore_svm = (2*master_precision_svm*master_recall_svm)/(master_precision_svm+master_recall_svm);

p_tree_user = [p_tree_user master_precision_tree]
r_tree_user = [r_tree_user master_recall_tree]
f_tree_user = [f_tree_user master_Fscore_tree]

p_svm_user = [p_svm_user master_precision_svm]
r_svm_user = [r_svm_user master_recall_svm]
f_svm_user = [f_svm_user master_Fscore_svm]
