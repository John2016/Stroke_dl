function [list_useful,list_delete]=filterConfuse(original_data)
%% ��֢״����һ�£��������һ�µ����ݹ��˵�
% Method: ������������һ�½����ͬ��������Ŀ
% dataΪ���ݵ㣬result�ǽ����ʾ�����ĳ��������data��ȫһ�£���result��һ�£���ô�������ݶ�ɾ��
%---------------------2014-9-25---------------------------------%
data = original_data(:,2:end);
result = original_data(:,1);
[n,m]=size(data);
count_d=0;
for i=1:n-1
    if sum(abs(data(i,:)))==0 %�����Ѿ���ɾ������Ŀ
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
            data(j,:)=zeros(1,m);% ���ظ���Ŀɾ��
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