function [list_useful,list_delete]=filterConfuse(original_data)
%% 将症状表现一致，但结果不一致的数据过滤掉
% Method: 消除两个表现一致结果不同的数据条目
% data为数据点，result是结果表示，如果某两行数据data完全一致，但result不一致，那么两行数据都删除
%---------------------2014-9-25---------------------------------%
data = original_data(:,2:end);
result = original_data(:,1);
[n,m]=size(data);
count_d=0;
for i=1:n-1
    if sum(abs(data(i,:)))==0 %跳过已经被删除的条目
        continue;
    end
    for j=i+1:n
        if sum(abs(data(j,:)))==0
            continue;
        end
        if sum(abs(data(i,:)-data(j,:)))==0&&result(j)~=result(i)
            count_d=count_d+1;
            list_delete(count_d,1)=i;
            list_delete(count_d,2)=j;
            data(j,:)=zeros(1,m);% 将重复条目删除
            data(i,:)=zeros(1,m);
            break;
        end
    end
end
list_deletel=unique(list_delete);
list=1:n;
list_useful=list;
list_useful(list_deletel)=[];
list_useful=list_useful';