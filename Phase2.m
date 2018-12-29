clc
close all
format longg

spoon_user9_emg = csvread('C:\Users\Simran Singh\Desktop\Data_Mining\MyoData\user28\spoon\EMG.txt');
spoon_user9_imu = csvread('C:\Users\Simran Singh\Desktop\Data_Mining\MyoData\user28\spoon\IMU.txt');

startend_user9 = csvread('C:\Users\Simran Singh\Desktop\Data_Mining\groundTruth\user28\spoon\timeframes.txt');
startend_user9_se = startend_user9( : ,(1:2));
startend9 = int64(startend_user9_se * (5/3));


emg_eating = [];
imu_eating = [];
emg_noneating = [];
imu_noneating = [];
for i=1:39    
    s = startend9(i,1);
    e = startend9(i,2);
    for j=2:9
        emg_eating(i,j-1) = min(spoon_user9_emg(s:e,j));
        emg_eating(i,j+7) = max(spoon_user9_emg(s:e,j));
        emg_eating(i,j+15) = rms(spoon_user9_emg(s:e,j));
        emg_eating(i,j+23) = mean(spoon_user9_emg(s:e,j));
        emg_eating(i,j+31) = min(spoon_user9_emg(s:e,j));
    end
    for j=2:11
        imu_eating(i,j-1) = min(spoon_user9_imu(s:e,j));
        imu_eating(i,j+9) = max(spoon_user9_imu(s:e,j));
        imu_eating(i,j+19) = rms(spoon_user9_imu(s:e,j));
        imu_eating(i,j+29) = mean(spoon_user9_imu(s:e,j));
        imu_eating(i,j+39) = min(spoon_user9_imu(s:e,j));
    end
    
    sne = e+1;
    ene = startend9(i+1,1)-1;
    for j=2:9
        emg_noneating(i,j-1) = min(spoon_user9_emg(sne:ene,j));
        emg_noneating(i,j+7) = max(spoon_user9_emg(sne:ene,j));
        emg_noneating(i,j+15) = rms(spoon_user9_emg(sne:ene,j));
        emg_noneating(i,j+23) = mean(spoon_user9_emg(sne:ene,j));
        emg_noneating(i,j+31) = min(spoon_user9_emg(sne:ene,j));
    end
    for j=2:11
        imu_noneating(i,j-1) = min(spoon_user9_imu(sne:ene,j));
        imu_noneating(i,j+9) = max(spoon_user9_imu(sne:ene,j));
        imu_noneating(i,j+19) = rms(spoon_user9_imu(sne:ene,j));
        imu_noneating(i,j+29) = mean(spoon_user9_imu(sne:ene,j));
        imu_noneating(i,j+39) = min(spoon_user9_imu(sne:ene,j));
    end
    
end

s = startend9(40,1);
e = startend9(40,2);

for j=2:9
    emg_eating(40,j-1) = min(spoon_user9_emg(s:e,j));
    emg_eating(40,j+7) = max(spoon_user9_emg(s:e,j));
    emg_eating(40,j+15) = rms(spoon_user9_emg(s:e,j));
    emg_eating(40,j+23) = mean(spoon_user9_emg(s:e,j));
    emg_eating(40,j+31) = min(spoon_user9_emg(s:e,j));
end
for j=2:11
    imu_eating(40,j-1) = min(spoon_user9_imu(s:e,j));
    imu_eating(40,j+9) = max(spoon_user9_imu(s:e,j));
    imu_eating(40,j+19) = rms(spoon_user9_imu(s:e,j));
    imu_eating(40,j+29) = mean(spoon_user9_imu(s:e,j));
    imu_eating(40,j+39) = min(spoon_user9_imu(s:e,j));  
end

user9_master = [];
user9_master((1:40),(1:40)) = emg_eating;
user9_master((1:40),(41:90)) = imu_eating;
user9_master((41:79),(1:40)) = emg_noneating;
user9_master((41:79),(41:90)) = imu_noneating;    

[coeff, scores, latent] = pca(user9_master);
Y_user9(1:40)=repmat(1,40,1);
Y_user9(41:79)=repmat(0,39,1);

Y_user9 = Y_user9'

tree = fitctree(scores([1:24,41:64], : ),Y_user9([1:24,41:64], : ));
label_tree = predict(tree,scores([25:40,65:79], : ));

C_tree = confusionmat(Y_user9([25:40,65:79], : )',label_tree);

recall_tree = C_tree(2,2)/sum(C_tree(2,:));
precision_tree = C_tree(2,2)/sum(C_tree(:,2));
Fscore_tree = (2*precision_tree*recall_tree)/(precision_tree+recall_tree);

svm1 = fitcsvm(scores([1:24,41:64], : ),Y_user9([1:24,41:64], : ))
label_svm = predict(svm1,scores([25:40,65:79], : ))

C_svm = confusionmat(Y_user9([25:40,65:79], : )',label_svm)

recall_svm = C_svm(2,2)/sum(C_svm(2,:));
precision_svm = C_svm(2,2)/sum(C_svm(:,2));
Fscore_svm = (2*precision_svm*recall_svm)/(precision_svm+recall_svm);

p_tree = [p_tree precision_tree]
r_tree = [r_tree recall_tree]
f_tree = [f_tree Fscore_tree]

p_svm= [p_svm precision_svm]
r_svm = [r_svm recall_svm]
f_svm = [f_svm Fscore_svm]


spoon_master = [spoon_master;user9_master];
Y_master = [Y_master;Y_user9];