clear;close all;
fp=imread('6_3.jpg');  
imshow(fp);
A=imresize(fp,[256,256]);
imwrite(A,'./result/?.jpg')
M0=80;  
Var0=200;
[len,wid]=size(fp);
fpn=zeros(len,wid);     
M=mean2(fp);             
Var=std2(fp)^2;             

fp=double(fp);   
for i=1:len
    for j=1:wid
        if fp(i,j)>M
            fpn(i,j)=M0+sqrt( Var0*(fp(i,j)-M)^2/Var );  
        else
            fpn(i,j)=M0-sqrt( Var0*(fp(i,j)-M)^2/Var );
        end
    end
end
fpn=uint8(fpn);
figure,imshow(fpn);

sobelx=[-1 0 1;-2 0 2;-1 0 1];
sobely=sobelx';
dx=zeros(len,wid);
dy=dx;
theta=dx;

for m=2:254
   for  n=2:254
       for i=1:3
           for j=1:3
              dx(m,n)=dx(m,n)+fp(m+i-2,n+j-2)*sobelx(i,j);
              dy(m,n)=dy(m,n)+fp(m+i-2,n+j-2)*sobely(i,j);
           end
       end
   end
end
     
block_dmap=ones(256,256);                

Vx=zeros(32,32);
Vy=Vx;
jiaodu=Vx;
for  i=1:32
  for       j=1:32                
       
       x=dx([1+(i-1)*8:8+(i-1)*8],[1+(j-1)*8:8+(j-1)*8]); 
       y=dy([1+(i-1)*8:8+(i-1)*8],[1+(j-1)*8:8+(j-1)*8]); 
       temp=x.*y;       
       Vx(i,j)=2*sum(temp(:));
       temp=x.^2-y.^2;
       Vy(i,j)=sum(temp(:));
       if Vy(i,j)==0
           jiaodu(i,j)=0;
       else
           jiaodu(i,j)=1/2*atan( Vx(i,j)/Vy(i,j) );
       theta([1+(i-1)*8:8+(i-1)*8],[1+(j-1)*8:8+(j-1)*8])=jiaodu(i,j);
       end

       theta([1+(i-1)*8:8+(i-1)*8],[1+(j-1)*8:8+(j-1)*8])=theta([1+(i-1)*8:8+(i-1)*8],[1+(j-1)*8:8+(j-1)*8])*180/pi;
       jiaodu(i,j)=jiaodu(i,j)*180/pi;
       if  Vy(i,j)>0
            jiaodu(i,j)=jiaodu(i,j)+90;
       end
       if   Vy(i,j)<0 & Vx(i,j)>0
            jiaodu(i,j)=jiaodu(i,j)+180;
       end

               if  jiaodu(i,j)<=11.25 |  jiaodu(i,j)>168.75;
                    y1=1;
                elseif    jiaodu(i,j)>11.25 &jiaodu(i,j)<=33.75;
                    y1=2;  
                elseif    jiaodu(i,j)>33.75 &  jiaodu(i,j)<=56.25; 
                    y1=3;  
                elseif   jiaodu(i,j)>56.25 & jiaodu(i,j)<=78.75;
                   y1=4;    
               elseif   jiaodu(i,j)>78.75 & jiaodu(i,j)<=101.25;
                    y1=5;
               elseif    jiaodu(i,j)>101.25 & jiaodu(i,j)<=123.75;
                    y1=6;
               elseif    jiaodu(i,j)>123.75 & jiaodu(i,j)<=146.25;
                    y1=7;
               elseif   jiaodu(i,j)>146.25 & jiaodu(i,j)<=168.75;
                    y1=8;
                end
      angle_xy=ones(8,8);
       
       switch  y1
                case 1
                  angle_xy(4,[2:7])=0;                 
                case 2
                  idx=sub2ind(size(angle_xy),[  3 4 4 5 5 6  ],[2:7]);
                  angle_xy(idx)=0;                 
                case 3
                  idx=sub2ind(size(angle_xy),[2:7],[2:7]);
                  angle_xy(idx)=0;
                case 4
                  idx=sub2ind(size(angle_xy),[2:7],[  3 4 4 5 5 6  ]);
                  angle_xy(idx)=0;
                case 5
                  angle_xy([2:7],4)=0;
                case 6
                  idx=sub2ind(size(angle_xy),[7:-1:2],[   3 4 4 5 5 6 ]);
                  angle_xy(idx)=0;
                case 7
                  idx=sub2ind(size(angle_xy),[7:-1:2],[7:-1:2]);
                  angle_xy(idx)=0;                     
                case 8
                  idx=sub2ind(size(angle_xy),[  6 5 5 4 4 3  ],[2:7]);
                  angle_xy(idx)=0;
                end
  
                block_dmap([1+(i-1)*8:8+(i-1)*8],[1+(j-1)*8:8+(j-1)*8])=angle_xy;
  end
end

               
figure;imshow(block_dmap);
imwrite(block_dmap,'./result/mask_?.jpg')
rever_block_dmap=1-block_dmap;              
rever_block_dmap=250*rever_block_dmap;     
addblock_dmap=imadd(fp,rever_block_dmap,'double');
figure;imshow(addblock_dmap,[]); 