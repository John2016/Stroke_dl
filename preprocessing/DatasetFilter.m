function [dataset_filtered_id]=DatasetFilter(dataset)
%% ���˵����ݿ�����Ч���ݵ���Ŀ
% ��Ч���ݱ��Ϊ-99
% �����Ч��������Ŀ��id
[rdata,cdata]=size(dataset);
idcount=0;
for i=1:rdata
    i
    flag=0;%��־�Ƿ������-99
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