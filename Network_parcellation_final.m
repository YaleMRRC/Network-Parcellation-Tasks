% Final code for Network level parcellation
%% Finding the subjects in common across all tasks


addpath(genpath('/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA'))
savepath

addpath(genpath('/home/mehraveh/documents/MATLAB/Parcellation/'))



filesRest1LR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/REST_LR/matrices/*_GSR_roimean.txt']);
filesRest1RL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/REST_RL/matrices/*_GSR_roimean.txt']);

filesRest2LR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/REST2_LR/matrices/*_GSR_roimean.txt']);
filesRest2RL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/REST2_RL/matrices/*_GSR_roimean.txt']);

filesMotLR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/MOTOR_LR/matrices/*_GSR_roimean.txt']);
filesMotRL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/MOTOR_RL/matrices/*_GSR_roimean.txt']);

filesGambLR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/GAMBLING_LR/matrices/*_GSR_roimean.txt']);
filesGambRL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/GAMBLING_RL/matrices/*_GSR_roimean.txt']);

filesWMLR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/WM_LR/matrices/*_GSR_roimean.txt']);
filesWMRL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/WM_RL/matrices/*_GSR_roimean.txt']);

filesEmLR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/EMOTION_LR/matrices/*_GSR_roimean.txt']);
filesEmRL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/EMOTION_RL/matrices/*_GSR_roimean.txt']);

filesLanLR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/LANGUAGE_LR/matrices/*_GSR_roimean.txt']);
filesLanRL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/LANGUAGE_RL/matrices/*_GSR_roimean.txt']);

filesRelLR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/RELATIONAL_LR/matrices/*_GSR_roimean.txt']);
filesRelRL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/RELATIONAL_RL/matrices/*_GSR_roimean.txt']);

filesSocLR = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/SOCIAL_LR/matrices/*_GSR_roimean.txt']);
filesSocRL = dir(['/mnt/store1/mridata2/mri_group/HCP_data/HCP_900_DATA/SOCIAL_RL/matrices/*_GSR_roimean.txt']);

files = {'filesMotLR','filesMotRL';'filesGambLR','filesGambRL'; ...
    'filesWMLR','filesWMRL';'filesEmLR','filesEmRL';'filesLanLR','filesLanRL';...
    'filesRelLR','filesRelRL';'filesSocLR','filesSocRL';'filesRest1LR',...
    'filesRest1RL';'filesRest2LR','filesRest2RL'}

%

for task = 1:9
    for lr=1:2
        subj_data=zeros(0,1);
        file_cur = eval(files{task,lr});
        for i = 1:length(file_cur)
            if (file_cur(i).bytes) ~= 0
                out=regexp(file_cur(i).name,'\d+','match');
                subj_data=[subj_data;str2double(cat(1,out{1}))];
            end
        end
        HCP_subj{task,lr}=subj_data;
    end
end


%  Find subjects in common among ALL TASKS

common = intersect(HCP_subj{1,1},HCP_subj{1,2});

for task = 1:9
    subjsTask = intersect(HCP_subj{task,1},HCP_subj{task,2});
    common = intersect(common,subjsTask);
end

for task = 1:9
    for lr = 1:2
        HCP_subj_common{task,lr} = arrayfun(@(x)find(HCP_subj{task,lr}==x,1),common);
    end
end



%% Every task with common global exemplars (718x9 subjects)
fprintf('3-Every task with common global exemplars (718x9 subjects) \n')


load HCP_data.mat
l_full = length(HCP_subj_common{1,1});

for task = 1:9
    task
    LR = HCP_subj_common{task,1};
    RL = HCP_subj_common{task,2};
    l_full = length(LR);
    
    for subj=1:l_full
%         subj
        V = [M{task,1}{LR(subj)}(:,cortical);M{task,2}{RL(subj)}(:,cortical)];

        t = size(V,1);
        n = size(V,2);
        mean_subtract = mean(V,2);
        V = V - repmat(mean_subtract,[1,n]);   

        twoNorm = sqrt(sum(abs(V).^2,1));
        m = max(twoNorm);
        V = V/m;

        sqDistances_HCP{task,subj} = sqDistance(V);

        D = sqDistances_HCP{task,subj};
        e0 = zeros(t,1);
        e0(1)=e0(1)+2.1;
        d0 = sqDistance_Y(V,e0);

        if min(d0)<=max(max(D(1:n,1:n)))
            fprintf('No :( \n')
        end
        
        d0_HCP{task, subj} = d0;
    end
end



clear Maj_all_tasks F_all_tasks index_global_all_tasks 

LR = HCP_subj_common{1,1};
RL = HCP_subj_common{1,2};
l_full = length(LR);

temp_sqDsitance=[];
temp_d0=[];
for task = 1:9    
    task
    temp_sqDsitance = [temp_sqDsitance, sqDistances_HCP(task,:)];
    temp_d0 = [temp_d0, d0_HCP(task,:)];    
end
   

S_opt = exemplar(temp_sqDsitance,temp_d0,t,n,50,l_full*9);

temp = S_opt;

for K=50:-1:2
    S_opt_all_all_tasks{K} = temp;
    temp=temp(1:end-1);
end

for task = 1:9
    for K=2:50  
        for subj = 1:l_full
            D = sqDistances_HCP{task,subj};
            ind = S_opt_all_all_tasks{K};
            [D_sorted,index_global_all_tasks{K,task}(subj,:)] = min(D(ind,1:end),[],1);
        end
        [Maj_all_tasks{K,task},F_all_tasks{K,task}] = mode(index_global_all_tasks{K,task});
    end            
end



%% number of nodes being added to each new network
clear all
load '/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/All_task_brainFigures/BrainFigures_task_1_9_WHOLE_BRAIN_global_exemplars_for_all_tasks_718x9subjs.mat'


for task = 1:9
    task
    for K=3:50
        for subj = 1:718
            A = index_global_all_tasks{K,task}(subj,:);
            B = index_global_all_tasks{K-1,task}(subj,:);
            new_network{task}(K,subj) = sum(A==K);  
            new_assignment{task}(K,subj) = sum(A~=B);
        end
    end
end


clear klabel
figure,

labels = {'Motor','Gambling','WM','Emotional','Language',...
    'Relational','Social','Rest1','Rest2'}    
for task=1:9
    C=new_assignment{task}';
%     C=new_network{task}';
% %     C=Hamm_dist_100{task}(:,2:50);
    subplot(3,3,task)
    boxplot(C)
    set(gca,'xtick',2:8:50,'xticklabel',2:8:50,'ytick',18:50:268,'yticklabel',18:50:268,'FontSize',18,'FontWeight','bold'); 
    ylim([1,268])
    xlim([1,52])
    format short e

    h = findobj(gca,'Type','line');
    set(h,'LineWidth',1)
    title([labels(task)], 'FontSize', 20)
    
    % Calculating the mean
%     C_mean(task,:)=mean(new_assignment{task}');
%     C_mean(task,:)=mean(new_network{task}');
%     stem(C_mean(task,:))
end

%% Looking at the variances using common global exemplars (718X9)

load /home/mehraveh/documents/MATLAB/Parcellation/2017/BrainFigures_task_1_9_cortical_global_exemplars_for_all_tasks_718x9subjs.mat
load /home/mehraveh/documents/MATLAB/Parcellation/HCP_distances.mat

clear Maj_all_tasks_400_400 F_all_tasks_400_400 index_global_all_tasks_400_400

l_full = 718;
l_half = floor(l_full/2);


for numberofruns=1:50
    numberofruns
    l_full_shuffled{numberofruns} = randperm(l_full);

    for task = 1:9
%         task
        for K=12
            i=0;
            for subj = l_full_shuffled{numberofruns}(1:l_half)
                i=i+1;
                D = sqDistances_HCP{task,subj};
                ind = S_opt_all_all_tasks{K};
                [D_sorted,index_global_all_tasks_400_400{numberofruns,1}{K,task}(i,:)] = min(D(ind,1:end),[],1);
            end
            [Maj_all_tasks_400_400{numberofruns,1}{K,task},F_all_tasks_400_400{numberofruns,1}{K,task}] = mode(index_global_all_tasks_400_400{numberofruns,1}{K,task});
            
            i=0;
            for subj = l_full_shuffled{numberofruns}(l_half+1:l_full)
                i=i+1;
                D = sqDistances_HCP{task,subj};
                ind = S_opt_all_all_tasks{K};
                [D_sorted,index_global_all_tasks_400_400{numberofruns,2}{K,task}(i,:)] = min(D(ind,1:end),[],1);
            end
            [Maj_all_tasks_400_400{numberofruns,2}{K,task},F_all_tasks_400_400{numberofruns,2}{K,task}] = mode(index_global_all_tasks_400_400{numberofruns,2}{K,task});
        end            
    end
end

% save Parcellation_400_400 Maj_all_tasks_400_400 F_all_tasks_400_400 index_global_all_tasks_400_400


clear Maj_all_tasks_400_400_K

for K=2:50
    for numberofruns=1:50
        for task = 1:9
            Maj_all_tasks_400_400_K{K,task}(numberofruns,:) = Maj_all_tasks_400_400{numberofruns,1}{K,task};Maj_all_tasks_400_400_K{K,task}(numberofruns+50,:) = Maj_all_tasks_400_400{numberofruns,2}{K,task};

        end
    end
end

%% Histograms with 400-400 split

% load('/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/2017/Parcellation_WHOLE_BRAIN_400_400_Ks.mat')
load('/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/2017/Maj_all_tasks_400_400_K_WHOLE_BRAIN.mat')
load ('/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/All_task_brainFigures/BrainFigures_task_1_9_WHOLE_BRAIN_global_exemplars_for_all_tasks_718x9subjs.mat')
addpath(genpath('/home/mehraveh/documents/MATLAB'))


numnodes = 268; %188 %268
numnodes_rectngle = 270; %192=12x16  %270=15x18
numnodes_rectngle_x = 18  %16 %18
numnodes_rectngle_y = 15  %12 %15


for K=[12] %[12,17,31,45]
    K
    clear Y
    for task = 1:9
        for p=1:numnodes
            [Y(p,task,:), x] = hist(Maj_all_tasks_400_400_K{K,task}(:,p),1:K);  
        end
        Y(numnodes+1:numnodes_rectngle,task,:) = 0;
    end



    clear l

    for k=1:K
        l{k} = num2str(k);
    end


    figure
    for t=1:numnodes_rectngle_y    % this is 188 = 12*16-4
        t
        subplot(numnodes_rectngle_y,1,t)
        indices = (t-1)*numnodes_rectngle_x+1:t*numnodes_rectngle_x;
    
        groupLabels = indices;
        plotBarStackGroups(Y(indices,:,:), groupLabels, 100); hold on % plot groups of stacked bars
        exemplar_indice = intersect(S_opt_all_all_tasks{K},indices)

        if K==12
            if ~isempty(exemplar_indice)
                [tf,loc]=ismember(exemplar_indice,indices);
                plot(loc,ones(length(loc),1)*50,'x', 'MarkerFaceColor','k',...
                'MarkerEdgeColor','k','MarkerSize',10,'linewidth',2)
            end
 
%             entropy_1_indice = intersect(find(y_pred==1),indices)
% 
%             if ~isempty(entropy_1_indice)
%                 [tf,loc]=ismember(entropy_1_indice,indices);
%                 plot(loc,ones(length(loc),1)*100,'.', 'MarkerFaceColor','r',...
%                 'MarkerEdgeColor','r','MarkerSize',30)
%             end
% 
% 
%             entropy_2_indice = intersect(find(y_pred==2),indices)
% 
%             if ~isempty(entropy_2_indice)
%                 [tf,loc]=ismember(entropy_2_indice,indices);
%                 plot(loc,ones(length(loc),1)*100,'.', 'MarkerFaceColor',[255,165,0]/255,...
%                 'MarkerEdgeColor',[255,165,0]/255,'MarkerSize',30)
%             end
% 
% 
%             entropy_3_indice = intersect(find(y_pred==3),indices)
% 
%             if ~isempty(entropy_3_indice)
%                 [tf,loc]=ismember(entropy_3_indice,indices);
%                 plot(loc,ones(length(loc),1)*100,'.', 'MarkerFaceColor','y',...
%                 'MarkerEdgeColor','y','MarkerSize',30)
%             end
% 
% 
% 
%             entropy_4_indice = intersect(find(y_pred==4),indices)
% 
%             if ~isempty(entropy_4_indice)
%                 [tf,loc]=ismember(entropy_4_indice,indices);
%                 plot(loc,ones(length(loc),1)*100,'.', 'MarkerFaceColor','g',...
%                 'MarkerEdgeColor','g','MarkerSize',30)
%             end
        end
% 
    end

end


%% Histograms with Maj Fs



load ('/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/All_task_brainFigures/BrainFigures_task_1_9_WHOLE_BRAIN_global_exemplars_for_all_tasks_718x9subjs.mat')

numnodes = 268; %188 %268
numnodes_rectngle = 270; %192=12x16  %270=15x18
numnodes_rectngle_x = 18  %16 %18
numnodes_rectngle_y = 15  %12 %15

for K=[12]
    K
    clear Y
for task = 1:9
    for p=1:numnodes
        Y(p,task,:) = hist(index_global_all_tasks{K,task}(:,p),1:K);  
%         temp = hist(Maj_all_tasks_400_400_K{K,task}(:,p),1:K);
%         consistencies(task,p) = max(temp)/200;
    end
    Y(numnodes+1:numnodes_rectngle,task,:) = 0;
end



clear l

for k=1:K
    l{k} = num2str(k);
end

% {'Mot','Gamb','WM','Em','Lan','Rel','Soc','Rest1','Rest2'};     % set labels

figure
for t=1:numnodes_rectngle_y      % this is 188 = 12*16-4
    t
    subplot(numnodes_rectngle_y,1,t)
    indices = (t-1)*numnodes_rectngle_x+1:t*numnodes_rectngle_x;
%     if t==11
%         indices=(t-1)*16+1:188;
%     end
    groupLabels = indices;
    plotBarStackGroups(Y(indices,:,:), groupLabels,718); % plot groups of stacked bars
end


legend(l)
end


%% Entropy within task and sum across tasks - Entropy y


clear Entropy_y Entropy_y_K

numnodes = 268; %188 %268
numnodes_rectngle = 270; %192=12x16  %270=15x18
numnodes_rectngle_x = 18  %16 %18
numnodes_rectngle_y = 15  %12 %15

kk=0;
for K = [12] %,24,39,42]
    kk=kk+1;
    clear Entropy_y
for p=1:numnodes
    for task = 1:9    
        [h, x] = hist(Maj_all_tasks_400_400_K{K,task}(:,p),1:K);
%         [h, x] = hist(index_global_all_tasks{K,task}(:,p),1:K);
        E = EntropyEstimationHist(h,x);        
        Entropy_y(p,task) = E;
        Entropy_y(numnodes+1:numnodes_rectngle,task) = 0;
        
    end
end

Entropy_y = sum(Entropy_y,2)/9;
Entropy_y=Entropy_y/max(Entropy_y) * 100;
Entropy_y_K(kk,:) = Entropy_y(1:numnodes);

figure
mat = reshape(1:numnodes_rectngle,numnodes_rectngle_x,numnodes_rectngle_y)';
mat2=reshape(Entropy_y,numnodes_rectngle_x,numnodes_rectngle_y)'
imagesc(reshape(Entropy_y,numnodes_rectngle_x,numnodes_rectngle_y)')
colorbar

textStrings = num2str(mat(:),'%i');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding

[x,y] = meshgrid(1:size(mat,2),1:size(mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center','fontsize',25);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
% textColors = repmat(mat(:) < midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color

textColors = double(repmat(mat2(:) < 0.8,1,3)); %.*repmat([0.2081 0.1663 0.5292],size(mat,1)*size(mat,2),1);
textColors([156,168,180,192],:) = repmat([0.2081 0.1663 0.5292],4,1);
% textColors([255,270],:) = repmat([0.2081 0.1663 0.5292],2,1);

set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors




% set(gca,'XTick',1:length(mat),...                         %# Change the axes tick marks
%         'XTickLabel',[all_label],...  %#   and tick labels
%         'YTick',1:length(mat),...
%         'YTickLabel',[all_label],...
%         'TickLength',[0 0],'Fontsize',25,'fontweight','bold');
% 
% 
end
%% Entropy within runtime (across tasks) and sum across runtimes - Entropy_x

clear Entropy_x Entropy_x_K
% 
% numnodes = 188; %188
% numnodes_rectngle = 192; %192=12x16  %270=15x18
% numnodes_rectngle_x = 16
% numnodes_rectngle_y = 12




for K=[12] %,24,39,42]
    for task=1:9
        for runtime=1:100 %718 %100
            for K=[12]
                for p=1:numnodes
                    Maj_all_tasks_400_400_K_reshaped{K,runtime}(task,p) = Maj_all_tasks_400_400_K{K,task}(runtime,p);
%                     index_global_all_tasks_reshaped{K,runtime}(task,p) = index_global_all_tasks{K,task}(runtime,p);
                end
            end
        end
    end
end

kk=0;
for K=[12] %,24,39,42]
    kk=kk+1;
for p=1:numnodes
    for runtime = 1:100    
        [h, x] = hist(Maj_all_tasks_400_400_K_reshaped{K,runtime}(:,p),1:K);
%         [h, x] = hist(index_global_all_tasks_reshaped{K,runtime}(:,p),1:K);
        E = EntropyEstimationHist(h,x);
        Entropy_x(p,runtime) = E;
        Entropy_x(numnodes+1:numnodes_rectngle,runtime) = 0;
    end
end
Entropy_x = sum(Entropy_x,2)/100
Entropy_x=Entropy_x/max(Entropy_x) *100;
Entropy_x_K(kk,:) = Entropy_x(1:numnodes);

figure
imagesc(reshape(Entropy_x,numnodes_rectngle_x,numnodes_rectngle_y)')
colorbar

mat = reshape(1:numnodes_rectngle,numnodes_rectngle_x,numnodes_rectngle_y)';
mat2=reshape(Entropy_x,numnodes_rectngle_x,numnodes_rectngle_y)'

textStrings = num2str(mat(:),'%i');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding

[x,y] = meshgrid(1:size(mat,2),1:size(mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center','fontsize',25);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
% textColors = repmat(mat(:) < midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color

textColors = double(repmat(mat2(:) < 0.8,1,3)); %.*repmat([0.2081 0.1663 0.5292],size(mat,1)*size(mat,2),1);
textColors([156,168,180,192],:) = repmat([0.2081 0.1663 0.5292],4,1);
% textColors([255,270],:) = repmat([0.2081 0.1663 0.5292],2,1);

set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors




% set(gca,'XTick',1:length(mat),...                         %# Change the axes tick marks
%         'XTickLabel',[all_label],...  %#   and tick labels
%         'YTick',1:length(mat),...
%         'YTickLabel',[all_label],...
%         'TickLength',[0 0],'Fontsize',25,'fontweight','bold');
% 
% 


end

%% Entropy_x versus Entropy_y with mean/medians as clustering as color
figure
subplot(1,2,1)
c = parula(4)

% x_median = mean(Entropy_x_K)
x_median= 0.0001;
y_median = mean(Entropy_y_K(Entropy_y_K>0.0001)) %median
y_pred=zeros(numnodes,1);


y_pred(Entropy_x_K>x_median & Entropy_y_K>y_median) = 4;
y_pred(Entropy_x_K>x_median & Entropy_y_K<y_median) = 3;
y_pred(Entropy_x_K<x_median & Entropy_y_K>y_median) = 2;
y_pred(Entropy_x_K<x_median & Entropy_y_K<y_median) = 1;


for p=1:numnodes
    plot(Entropy_x_K(p),Entropy_y_K(p),'.', 'color',c(y_pred(p),:),'markersize',40), hold on
    if y_pred(p)==1
        plot(Entropy_x_K(p),Entropy_y_K(p),'r.','markersize',40), hold on
    elseif y_pred(p)==2
        plot(Entropy_x_K(p),Entropy_y_K(p),'.', 'color',[255,165,0]/255,'markersize',40), hold on
    elseif y_pred(p)==3
        plot(Entropy_x_K(p),Entropy_y_K(p),'y.','markersize',40), hold on
    elseif y_pred(p)==4
        plot(Entropy_x_K(p),Entropy_y_K(p),'g.','markersize',40), hold on
    end   
    
    
    
    
    
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', 16)
    
end
plot([0 100],[y_median y_median], 'r-','linewidth',4)
plot([x_median x_median],[0 100], 'r-', 'linewidth',4)
% colorbar 
xlabel ('Cross-task entropy','FontSize',20,'FontWeight','bold');
ylabel('Within-task entropy','FontSize',20,'FontWeight','bold');
title('Mean','FontSize',20,'FontWeight','bold');
% set(gca, 'xlim', [-0.1,1.1])
% set(gca, 'ylim', [-0.1,1.1])

% y_pred(S_opt_all_all_tasks{K})=5


%% Entropy_x versus Entropy_y FOR EACH NETWORK SEPARATELY
% load '/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/All_task_brainFigures/BrainFigures_task_1_9_cortical_global_exemplars_for_all_tasks_718x9subjs.mat'
load '/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/All_task_brainFigures/BrainFigures_task_1_9_WHOLE_BRAIN_global_exemplars_for_all_tasks_718x9subjs.mat'
K=12
% labels={'Motor','Gambling','Working Memory','Emotional','Language',...
%     'Relational','Social','Rest1','Rest2'};     % set labels
figure
c = parula(K);
task=8;
% mixed_x_y = ceil(((Entropy_x_K/100)+(Entropy_y_K/100)*3)*10);
% c = parula(max(mixed_x_y));

numnodes = 268; %188 %268
numnodes_rectngle = 270; %192=12x16  %270=15x18
numnodes_rectngle_x = 18  %16 %18
numnodes_rectngle_y = 15  %12 %15

% x_median = mean(Entropy_x_K)
x_median= 0.001;
y_median = mean(Entropy_y_K(Entropy_y_K>0.0001))
y_pred=zeros(numnodes,1);


y_pred(Entropy_x_K>x_median & Entropy_y_K>y_median) = 4;
y_pred(Entropy_x_K>x_median & Entropy_y_K<y_median) = 3;
y_pred(Entropy_x_K<x_median & Entropy_y_K>y_median) = 2;
y_pred(Entropy_x_K<x_median & Entropy_y_K<y_median) = 1;


labels={'Ventral Attention';'Dorsal Attention';'Language';'Vision II';
        'DMN (Dorsal-medial)';'Cingulo-opercular';'Single node!';'DMN (Core and Medial-temporal)';
        'Vision I';'Fronto-parietal';'Sensorimotor';'Subcortical-Cerebellum'}


for networks = 1:K
    p_networks{networks} = find(Maj_all_tasks{12,8}==networks);
end
    
    
for networks=1:12
    networks
    subplot(3,4,networks)
    
    hAnnot(networks) = annotation('textbox', [mod(networks-1,4)*0.22+0.13, 0.65-floor((networks-1)/4)*.3 .1 .1],...
    'String', [num2str(sum(y_pred(p_networks{networks})==1))],'FontSize',20,'FontWeight','bold','LineStyle','none','color','r');

    hAnnot(networks) = annotation('textbox', [mod(networks-1,4)*0.22+0.25, 0.65-floor((networks-1)/4)*.3 .1 .1],...
    'String', [num2str(sum(y_pred(p_networks{networks})==3))],'FontSize',20,'FontWeight','bold','LineStyle','none','color','y');

    hAnnot(networks) = annotation('textbox', [mod(networks-1,4)*0.22+0.13, 0.82-floor((networks-1)/4)*.3 .1 .1],...
    'String', [num2str(sum(y_pred(p_networks{networks})==4))],'FontSize',20,'FontWeight','bold','LineStyle','none','color','g');

    

    for p=p_networks{networks}
        
        if y_pred(p)==1
            plot(Entropy_x_K(p),Entropy_y_K(p),'r.','markersize',40), hold on
            
            
            
        elseif y_pred(p)==2
            plot(Entropy_x_K(p),Entropy_y_K(p),'.', 'color',[255,165,0]/255,'markersize',40), hold on
        elseif y_pred(p)==3
            plot(Entropy_x_K(p),Entropy_y_K(p),'y.','markersize',40), hold on
             
            
        elseif y_pred(p)==4
            plot(Entropy_x_K(p),Entropy_y_K(p),'g.','markersize',40), hold on
             
             
        end   
        
        xt = get(gca, 'XTick');
        set(gca, 'FontSize', 16)
        
        title([labels{networks}])
        
        if networks == 10
            xlabel ('Cross-task entropy','FontSize',20,'FontWeight','bold');
        end
        if networks == 5
            ylabel('Within-task entropy','FontSize',20,'FontWeight','bold');
        end
%         set(gca, 'xlim', [-0.1,1.1]*100)
%         set(gca, 'ylim', [-0.1,1.1]*100)
        
            %// Customize position here
      

    end
    plot([0 100],[y_median y_median], 'r-','linewidth',4)
    plot([x_median x_median],[0 100], 'r-', 'linewidth',4)

end



%% Histograms of 3classes of entropies in networks
load /Users/Mehraveh/Documents/MATLAB/Parcellation/2017/All_task_brainFigures/BrainFigures_task_1_9_WHOLE_BRAIN_global_exemplars_for_all_tasks_718x9subjs

labels={'VAN';'DAN';'Language';'Vis II';
        'DMN (Dorsomedial)';'Cing-Op';'-';'DMN (Core)';
        'Vis I';'FPN';'SMN';'Sub-Cereb'}
K=12
figure, 
C = categorical(Maj_all_tasks{12,8}(find(y_pred==1)),1:12,labels)
subplot(141)
c=[[255,0,0];[0,206,209];[255,255,0];[0,255,0];[0,0,139];[0,134,139];...
    [58,95,205];[255,165,0];[0,100,0];[205,41,144];[255,218,185];[239,128,128]]./255;
h=histogram(C,'edgecolor','r','facecolor','r')

V=h.Values';
xt = get(gca, 'XTick');
set(gca,'XTickLabelRotation',45)
set(gca, 'FontSize', 20)
set(gca, 'FontWeight', 'bold')
% set(gca, 'xlim', [0.5,12.5])
title([{'Low cross-task var'}; {'Low cross-subj var'}])

C = categorical(Maj_all_tasks{12,8}(find(y_pred==3)),1:12,labels)
subplot(142)
histogram(C,'edgecolor','y','facecolor','y')
xt = get(gca, 'XTick');
set(gca,'XTickLabelRotation',45)
set(gca, 'FontSize', 16)
set(gca, 'FontWeight', 'bold')
% set(gca, 'xlim', [0.5,12.5])

title([{'High cross-task var'}; {'Low cross-subj var'}])

C = categorical(Maj_all_tasks{12,8}(find(y_pred==4)),1:12,labels)
subplot(143)
histogram(C,'edgecolor','g','facecolor','g')
set(gca, 'XTick');
set(gca,'XTickLabelRotation',45)
set(gca, 'FontSize', 16)
set(gca, 'FontWeight', 'bold')
% set(gca, 'xlim', [0.5,12.5])
title([{'High cross-task var'}; {'High cross-subj var'}])


C = categorical(Maj_all_tasks{12,8},1:12,labels)
subplot(144)
histogram(C,'edgecolor','k','facecolor','k')
xt = get(gca, 'XTick');
set(gca,'XTickLabelRotation',45)
set(gca, 'FontSize', 16)
set(gca, 'FontWeight', 'bold')
% set(gca, 'xlim', [0.5,12.5])
title([{'Random dist.'}])

%% piechart FOR EACH NETWORK SEPARATELY
figure
labels={'VAN';'DAN';'Language';'Vis II';
        'DMN (Dorsomedial)';'Cing-Op';'-';'DMN (Core)';
        'Vis I';'FPN';'SMN';'Sub-Cereb'}
for K = 1:12
C = categorical(y_pred(find(Maj_all_tasks{12,8}==K)),[1,3,4])
subplot(3,4,K)

h=pie(C)
c=[[255,0,0];[255,255,0];[0,255,0]]./255;

hp = findobj(h, 'Type', 'patch');
% if K==4
%     c(3,:)=[];
% end
if K==7
    c(2:3,:)=[];
end

for i = 1:length(hp)
    set(hp(i), 'FaceColor', c(i,:));        
    set(hp(i), 'EdgeColor', c(i,:));        
end
hp = findobj(h, 'Type', 'Text');
for i = 1:length(hp)
    set(hp(i), 'FontSize', 25);
    out=regexp(get(hp(i), 'String'),'\d+%','match');
    inside=1; 
    set(hp(i), 'String',out)
    set(hp(i), 'String','')
end

title(labels{K},'fontsize',20)
end


%% pie chart of 3classes of entropies in networks
load /Users/Mehraveh/Documents/MATLAB/Parcellation/2017/All_task_brainFigures/BrainFigures_task_1_9_WHOLE_BRAIN_global_exemplars_for_all_tasks_718x9subjs

labels={'VAN';'DAN';'Language';'Vis II';
        'DMN (Dorsomedial)';'Cing-Op';'-';'DMN (Core)';
        'Vis I';'FPN';'SMN';'Sub-Cereb'}
K=12
figure, 
C = categorical(Maj_all_tasks{12,8}(find(y_pred==1)),1:12,labels)
subplot(131)
c=[[255,0,0];[0,206,209];[255,255,0];[0,255,0];[0,0,139];[0,134,139];...
    [58,95,205];[255,165,0];[0,100,0];[205,41,144];[255,218,185];[239,128,128]]./255;
h=pie(C,labels)

hp = findobj(h, 'Type', 'text');
for i=1:K
    ismember(labels{i},hp(i).String)
end

for i = 1:K        
    set(hp(i), 'FontSize', 25);
    out=regexp(get(hp(i), 'String'),'\d+%','match');
    inside=1; 
    set(hp(i), 'String',out)    
    set(hp(i), 'String','')
end



hp = findobj(h, 'Type', 'patch');
for i = 1:K        
    set(hp(i), 'FaceColor', c(i,:)); 
    set(hp(i), 'EdgeColor', c(i,:)); 
end



title([{'Low cross-task var'}; {'Low cross-subj var'}],'fontsize',20)

C = categorical(Maj_all_tasks{12,8}(find(y_pred==3)),1:12,labels)
% This doesn't have the single node. Thus, it is 11 networks.

subplot(132)

h=pie(C,labels);
hp = findobj(h, 'Type', 'text');

for i = 1:11 
    set(hp(i), 'FontSize', 25);
    out=regexp(get(hp(i), 'String'),'\d+%','match');
    inside=1; 
    set(hp(i), 'String',out)    
    set(hp(i), 'String','')
end
hp = findobj(h, 'Type', 'patch');
c_11=c;
c_11(7,:)=[];

for i = 1:11
    set(hp(i), 'FaceColor', c_11(i,:)); 
    set(hp(i), 'EdgeColor', c_11(i,:)); 
end


title([{'High cross-task var'}; {'Low cross-subj var'}],'fontsize',20)

C = categorical(Maj_all_tasks{12,8}(find(y_pred==4)),1:12,labels)
% Doesn't have vis II and Single node
subplot(133)
% explode = {};
h=pie(C,labels)
hp = findobj(h, 'Type', 'text');

for i = 1:10       
    set(hp(i), 'FontSize', 25);
    out=regexp(get(hp(i), 'String'),'\d+%','match');
    inside=1; 
    set(hp(i), 'String',out)
    set(hp(i), 'String','')
end

c_10=c;
c_10([7],:)=[];


hp = findobj(h, 'Type', 'patch');
for i = 1:11      
    set(hp(i), 'FaceColor', c_10(i,:));
    set(hp(i), 'EdgeColor', c_10(i,:)); 
end

title([{'High cross-task var'}; {'High cross-subj var'}],'fontsize',20)


%% Reading As from R for Red, Yellow, and Green connections' comparison
clear meansA stdsA pvalsA
for task=1:9
    load(['/Users/Mehraveh/Documents/MATLAB/Parcellation/2017/2017/As_fromR_Task',num2str(task),'.mat']);    
    meansA(task,1) = mean(A1);
    meansA(task,2) = mean(A2);
    meansA(task,3) = mean(A3);
    meansA(task,4) = mean(A4);
    meansA(task,5) = mean(A5);
    meansA(task,6) = mean(A6);
    stdsA(task,1) = std(A1);
    stdsA(task,2) = std(A2);
    stdsA(task,3) = std(A3);
    stdsA(task,4) = std(A4);
    stdsA(task,5) = std(A5);
    stdsA(task,6) = std(A6);
    pvalsA(task,:) = pval(:);
end



labels={'red,red','yellow,yellow','green,green','red,yellow','reg,green','yellow,green'}
labels={'r,r','y,y','g,g','r,y','r,g','y,g'}
task_labels = {'Motor','Gambling','WM','Emotional','Language',...
    'Relational','Social','Rest1','Rest2'}    
c = [[255,23,25];[255,242,0];[72,239,2];[255,127,14];[72,113,63];[23,190,207]]./255;
c = [[255,23,25];[255,242,0];[72,239,2];[100, 100, 100];[150, 150, 150];[204, 204, 204]]./255;

% figure
groups={[2,3],[2,4],[2,6],[3,5],[3,6],[4,5],[4,6],[5,6]}; %[1,2],[1,3],[1,4],[1,5],
pvalue_ind = [7,8,10,14,15,19,20,25]; %1,2,3,4,
% figure

for task=1:9
figure('rend','painters','pos',[100 100 500 500])
subtightplot(1,1,1,[0.01,0.01],0.075,0.075)

barData=meansA(task,:);

for i=1:6
    H=bar(i,barData(i)), hold on
    set(H,'FaceColor',c(i,:),'Edgecolor',c(i,:))
    errorbar(i,barData(i),stdsA(task,i),'.','Color',c(i,:),'linewidth',4,'CapSize',10)        
    
end
set(gca,'FontSize',20)
xlim([0.5,6.5])

ylim([0.1,0.7])
set(gca,'Xtick',1:6)
set(gca,'XtickLabel',labels)
% set(gca,'XTickLabelRotation',45)
% axis('square','xy');


% figure
errorbar(i,barData(i),stdsA(task,i),'.','Color',c(i,:),'linewidth',4)        
%overlay random data

[H,starY]=sigstar(groups,pvalsA(task,pvalue_ind));


text(1,max(starY)+0.01,'****',...
        'HorizontalAlignment','Center',...
        'BackGroundColor','none',...
        'Tag','sigstar_stars','Fontsize',25);
title([task_labels{task}],'fontsize',20);
saveas(gcf,['/Users/Mehraveh/Desktop/RYG_strength_Task',num2str(task),'.tif'])
end


