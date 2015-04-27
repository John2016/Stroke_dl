function [dataset_filtered_id]=DatasetFilter(dataset)
%% 过滤掉数据库中无效数据的项目
% 无效数据被填补为-99
% 输出有效的数据条目的id
[rdata,cdata]=size(dataset);
idcount=0;
for i=1:rdata
    i
    flag=0;%标志是否检索到-99
    for j=1:cdata
        if dataset(i,j)==-99
            flag=1;
            break;
        end
    end
    if flag==0
        idcount=idcount+1;
        dataset_filtered_id(idcount)=i;
    end
end