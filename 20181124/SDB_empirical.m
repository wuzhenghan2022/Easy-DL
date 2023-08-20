% Program to develop a relation between insitu depths and SDB method
% (Stumpf 2003) and Lyzenga models
% it requires netcdf Level 2 data in netcdf format derived from ACOLITE software
% for processing multiple satellite images and compositing technique. Works
% for both Sentinel-2 A and 2 B satellites

% For better mapviewing, use m_map toolbox, download from
% (https://www.eoas.ubc.ca/~rich/map.html)

% ACOLITE atmospheric correction software can be obtained from
% (https://odnature.naturalsciences.be/remsem/software-and-data/acolite)


clc
clear

% change the paths according to your data
% addpath('F:\softwares\m_map1.4\m_map');
% addpath('D:\SATELLITE_DERIVED_BATHYMETRY');
% addpath('D:\SATELLITE_DERIVED_BATHYMETRY\GIS');
% addpath('D:\SATELLITE_DERIVED_BATHYMETRY\worlddatamap');
% addpath('D:\SATELLITE_DERIVED_BATHYMETRY\geoimread');
% addpath('D:\SATELLITE_DERIVED_BATHYMETRY\DrosteEffect-BrewerMap-221b913\DrosteEffect-BrewerMap-221b913');
% addpath('F:\softwares\taylordiagram');
% addpath('F:\softwares\heatscatter');


%uncomment the filename you want to process 
fname = 'S2B_MSI_2018_10_05_04_59_33_T44QQE_L2W.nc';
%fname = 'S2A_MSI_2018_10_20_04_48_21_T44QQE_L2W.nc';
%fname = 'S2B_MSI_2018_10_25_04_48_49_T44QQE_L2W.nc';
%fname = 'S2A_MSI_2018_10_30_04_49_21_T44QQE_L2W.nc';
%fname = 'S2A_MSI_2018_11_19_04_51_01_T44QQE_L2W.nc';
%fname = 'S2B_MSI_2018_11_24_04_51_19_T44QQE_L2W.nc';


% You can specify the limits of your AOI
%Rushikonda limits
region = 'Rushi_Beach';
LONLIMS = [83.37,83.405];
LATLIMS = [17.765,17.80];

% Place all the processed ACOLITE L2W files in one folder and run

if strcmp(fname(1),'L')
    sat = fname(1:2);
    dat = fname(8:17);
    dat1 = fname(8:11);
    dat2 = fname(13:14);
    dat3 = fname(16:17);
    dat = [dat1,'-',dat2,'-',dat3];
    
else
    sat = fname(1:3);
    dat = fname(9:18);

end

% common bands
b2 = 'Rrs_492';
b4 = 'Rrs_665';  
b5 = 'Rrs_704';
b6 = 'Rrs_739';
b7 = 'Rrs_780';
b8 = 'Rrs_833';
b11 = 'Rrs_2186';

if strcmp(sat,'S2A')
    b1 = 'Rrs_443';
    b3 = 'Rrs_560';
     b6 = 'Rrs_740';
     b7 = 'Rrs_783';
    b9 = 'Rrs_865';
    b10 = 'Rrs_1614';
    b11 = 'Rrs_2202';
elseif strcmp(sat,'S2B')
    b1 = 'Rrs_442';
    b3 = 'Rrs_559';  
    b9 = 'Rrs_864';
    b10 = 'Rrs_1610';
    b11 = 'Rrs_2186';
else
     b1 = 'Rrs_443';
     b2 = 'Rrs_483';
     b3 = 'Rrs_561';
     b4 = 'Rrs_655';
     b5 = 'Rrs_704';
     b6 = 'Rrs_739';
     b7 = 'Rrs_780';
     b8 = 'Rrs_865';
end


%ncdisp(fname);
lat = ncread(fname,'lat');
lon = ncread(fname,'lon');

[rr1,cc1] = find(lat>=LATLIMS(1,1) & lat<=LATLIMS(1,2) & lon>=LONLIMS(1,1)& lon<=LONLIMS(1,2));

lon = ncread(fname,'lon',[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
lat = ncread(fname,'lat',[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);

r1 = ncread(fname,b1,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r2 = ncread(fname,b2,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r3 = ncread(fname,b3,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r4 = ncread(fname,b4,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r5 = ncread(fname,b5,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r6 = ncread(fname,b6,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r7 = ncread(fname,b7,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r8 = ncread(fname,b8,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r9 = ncread(fname,b9,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r10 = ncread(fname,b10,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
r11= ncread(fname,b11,[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);

% spm = ncread(fname,'spm_nechad2016',[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
% turb = ncread(fname,'t_nechad2016',[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
% 
% chl = ncread(fname,'chl_oc2',[min(rr1) min(cc1)],[max(rr1)-min(rr1) max(cc1)-min(cc1)]);
% spm = double(spm); chl = double(chl); turb = double(turb);
% spm = medfilt2(spm, [3 3]); chl = medfilt2(chl, [3 3]); turb = medfilt2(turb, [3 3]); 
% 

r1= double(r1); r2 = double(r2); r3 = double(r3); r4 = double(r4); 
r5 = double(r5); r6 = double(r6); r7 = double(r7); r8 = double(r8); 
r9 = double(r9); r10 = double(r10); r11 = double(r11); 

r1 = medfilt2(r1, [3 3]); r2 = medfilt2(r2, [3 3]); r3 = medfilt2(r3, [3 3]); 
r4 = medfilt2(r4, [3 3]); r5 = medfilt2(r5, [3 3]); r6 = medfilt2(r6, [3 3]); 
r7 = medfilt2(r7, [3 3]); r8 = medfilt2(r8, [3 3]); r9 = medfilt2(r9, [3 3]); 
r10 = medfilt2(r10, [3 3]); r11 = medfilt2(r11, [3 3]);


for i=1:11
eval(['aa(i,1)','=','r', num2str(i),'(173,195)',';']);
eval(['bb(i,1)','=','r', num2str(i),'(349,257)',';']);
eval(['cc(i,1)','=','r', num2str(i),'(45,347)',';']);
eval(['dd(i,1)','=','r', num2str(i),'(139,246)',';']);
end

figure(1)
plot(aa,'-','LineWidth',2,'color',[146 36 40]./255)
hold on
plot(bb,'-','LineWidth',2,'color',[144 147 203]./255)
hold on
plot(cc,'-','LineWidth',2,'color',[215 185 99]./255)
hold on
plot(dd,'-','LineWidth',2,'color',[132 186 91]./255)
ylim([0 0.15])
dt = dat(9:10); mn = dat(6:7); yr = dat(1:4);

title([dt,'-',mn,'-',yr],'FontSize',14)

xlabel('S2 Band Number','FontSize',14)
ylabel('Rrs (sr^-^1)','FontSize',14)
xticklabels({'b1','b2','b3','b4','b5','b6','b7','b8','b8A','b11','b12'})
legend('coastal','deepwater','sand','grass','Location','northwest')
set(gca,'FontSize',12)
fig = [sat,'_',region,'_Rrs_',dat,'.tif'];
print(1,'-dtiff',fig,'-r600');

%%
lon = double(lon); lat = double(lat);

NDWI = (r3-r8)./(r3+r8); % correct formula for water bodies (green-nir)/(green+nir)
mask = NDWI;

mask(mask>=0)=1;
mask(mask<0)=NaN;
lonm=lon.*mask;
latm=lat.*mask;
r1 = r1.*mask; 
r2 = r2.*mask; 
r3 = r3.*mask; 
r4 = r4.*mask; 
r5 = r5.*mask; 
r6 = r6.*mask; 
r7 = r7.*mask; 
r8 = r8.*mask; 
r9 = r9.*mask;
r10 = r10.*mask;
r11 = r11.*mask;
r12=lonm;
r13=latm;
%%

for i=1:13
eval(['r',num2str(i),'a = reshape(r',num2str(i),',376*391,1);']);
end

cr = zeros(13,13);
for i=1:11
    for j=1:11
        
        formatSpec = "corr(r%sa,r%sa,'rows','complete');";
        str = sprintf(formatSpec,num2str(i),num2str(j));
        cr(i,j) = eval(str);
    end
end


%% Stumpf method for SDB (Blue/Green)
a = log(1000*pi*r2); % blue
b = log(1000*pi*r3); % green
c = a./b;
Cbg = c; %medfilt2(c, [3 3]); % median filter

% % Green/Red
% clear a b c;
% a = log(1000*pi*r3); % green
% b = log(1000*pi*r4); % red
% c = a./b;
% Cgr = c; %medfilt2(c, [3 3]); % median filter
% %C= c;
% 
% % Blue / Red
% clear a b c;
% a = log(1000*pi*r2); % blue
% b = log(1000*pi*r4); % red
% c = a./b;
% Cbr = c; %medfilt2(c, [3 3]); % median filter

% Lyzenga method for SDB

xi = log(r1-min(min(r1)));
xj = log(r2-min(min(r2)));
xk = log(r3-min(min(r3)));
xl = log(r4-min(min(r4)));
xm = log(r5-min(min(r5)));
xn = log(r6-min(min(r6)));
xo = log(r8-min(min(r8)));
xp = log(r10-min(min(r10)));
%xq = log(r11-min(min(r11)));


%% Importing reference depth for algorithm development

% depth = importdepth('Rushi_validation.csv');
csv = csvread('Rushi_validation.csv');
depth=csv;
[m,n] = size(depth) ;
P = 0.8 ; % percentage of training dataset
idx = randperm(m)  ; % randomly selects the data
Training = depth(idx(1:round(P*m)),:) ;     % training data
Testing = depth(idx(round(P*m)+1:end),:) ;  % testing data
% lt = Training.lt;
% ln = Training.ln;
% dep = Training.d;
lt = Training(:,2);
ln = Training(:,1);
dep = Training(:,3);
dep = round(dep,3);
dep(dep==0)=0.01;
%dep = dep.*-1;
% Training.d = dep;

ln(dep<0)=NaN;
lt(dep<0)=NaN;
dep(dep<0)=NaN;
lt(isnan(lt))='';
ln(isnan(ln))='';
dep(isnan(dep))=''; %dep=dep*-1;

[ss,~] = size(dep);

win = 4.5e-4; % 50 m window
ln1 = zeros(ss,1); lt1= ln1; C1 = ln1; C2 = ln1; C3 = ln1; XI = ln1; XJ = ln1; XK=ln1;
XL = ln1;XM = ln1;XN = ln1;XO = ln1;XP = ln1;XQ = ln1;

disp('Please wait .. finding closest data')

tic
for k=1:ss
disp([num2str(k) ' out of ' num2str(ss)])

if ln(k)> min(min(lon)) && ln(k)< max(max(lon))
if lt(k)> min(min(lat)) && lt(k)< max(max(lat))
    cbg = Cbg(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
%   cbr = Cbr(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
%   cgr = Cgr(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xii = xi(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xjj = xj(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xkk = xk(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xll = xl(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xmm = xm(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xnn = xn(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xoo = xo(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    xpp = xp(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
        
    lnn = lon(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    ltt = lat(lon<ln(k)+win & lon>ln(k)-win & lat<lt(k)+win & lat>lt(k)-win);
    dist = sqrt((ln(k)-lnn).^2+(lt(k)-ltt).^2);
    md = (min((dist)));
    [row,col] = find(dist==md);
    [si,sj] = size(col);
    if si >= 1 && sj>=1
        row1 = min(row);
        col1 = min(col);
        clear col;
        clear row;
        col = col1;
        row = row1;
    
        ln1(k) = lnn(row,col);
        lt1(k) = ltt(row,col);
        C1(k) = cbg(row,col);
%       C2(k) = cbr(row,col);
%       C3(k) = cgr(row,col);
        XI(k) = xii(row,col);
        XJ(k) = xjj(row,col);
        XK(k) = xkk(row,col);
        XL(k) = xll(row,col);
        XM(k) = xmm(row,col);
        XN(k) = xnn(row,col);
        XO(k) = xoo(row,col);
        XP(k) = xpp(row,col);
    
    end 
end
end
end
toc 

data=[ln1,lt1,dep,XI,XJ,XK,XL,XM,XN,XO,XP];


%% Lazenga model full
XI(XI==0| XI == -Inf | XI == Inf)=NaN;
XJ(XJ==0| XJ == -Inf | XJ == Inf)=NaN;
XK(XK==0| XK == -Inf | XK == Inf)=NaN;
XL(XL==0| XL == -Inf | XL == Inf)=NaN;
XM(XM==0| XM == -Inf | XM == Inf)=NaN;
XN(XN==0| XN == -Inf | XN == Inf)=NaN;
XO(XO==0| XO == -Inf | XO == Inf)=NaN;
XP(XP==0 | XP == -Inf | XP == Inf)=NaN;

XX = [ones(size(XI)) XI XJ XK XL XM XN XO];% XP];
reg = regress(dep,XX);    % Removes NaN data

a0 = reg(1);
ai = reg(2);
aj = reg(3);
ak = reg(4);
al = reg(5);
am = reg(6);
an = reg(7);
ao = reg(8);

ZL = a0 + ai*XI + aj*XJ + ak*XK + al*XL + am*XM + an*XN + ao*XO;

disp('Lyzenga selected var');
   R = corr(dep,ZL,'rows','complete')
    RR2 = R*R

ZLm = a0 + ai*xi + aj*xj + ak*xk + al*xl + am*xm + an*xn + ao*xo;% + ap*xp;
 
% figure
% axesm miller
% pcolorm(lat,lon,ZLm)
% caxis([0 15])
% colorbar

%% for Stumpf model
    clc
    C1(C1==0)=NaN;
%   C2(C2==0)=NaN;
%   C3(C3==0)=NaN;
   
    C11 = [ones(size(C1)) C1]; % blue -green
    reg = regress(dep,C11);    % Removes NaN data
    
    a0 = reg(1);
    a1 = reg(2);
    
    ZSbg = a0 + a1*C1;
    
    
    R = corr(dep,C1,'rows','complete')
    RR2 = R*R
    disp(['R2 (blue-green) = ', num2str(RR2)])

    
    disp(['SDB_Stumpf(bg) = ',num2str(round(a1,2)),' x pSDB_bg ',num2str(round(a0,2))])
    
    SDB_Stumpf_bg = a0 + a1*Cbg; % Stump model original
    
%%
    
    disp('For Stumpf...')
    val1 = dep;
    val2 = ZSbg;
    
    val1(isnan(val2))= '';
    val2(isnan(val2))= '';
    
dd1 = (val1-val2);
dd3 = (val1-val2)./val2;
dd = (val1-val2).^2;
[mk,~]=size(dd);


% statistics computation
R = corr(val1,val2,'rows','complete');
RR2 = R*R
bias = sum(dd1)/mk % also known as Mean prediction difference (MPD)
RMSE = sqrt(sum(dd)/mk) % also known as Root mean sq pred diff (RMSPD)

%STDE = sqrt(sum((dd1 - ubias).^2)/mk)
dd2 = abs(dd1); 
MAE = sum(dd2)/mk
umsat = mean(val1);
umins = mean(val2);
udiff = umsat-umins;
uper = (abs(udiff)/umins)*100;
ust1 = std(val1);
ust2 = std(val2);
MNB = sum(dd3)/mk % Mean normalised bias
diffmed = median(val1)-median(val2)
mk

%%

figure(2)

SS = shaperead('vskp_watermask.shp');
isin = inpolygon(lon,lat,SS.X,SS.Y);

SDB_Stumpf_bg(isin) = NaN;

if strcmp(region,'Rushi_Beach')
    S = shaperead('vskp_shoreline-line.shp');
end


m_proj('Transverse Mercator','lon',LONLIMS,'lat',LATLIMS);
[XX,YY] = m_ll2xy(S.X,S.Y);
m_pcolor(lon,lat,SDB_Stumpf_bg)
shading 'interp';
hold on
geoshow(YY,XX,'Color','black');
colormap(flipud(parula));
%colormap(flipud(brewermap([],'*RdBu')));
m_grid('linewidth',2,'tickdir','out','tickstyle','dm','xtick',5,'ytick',5,'fontsize',18);
set(gca,'FontSize',30)
dat1 = [num2str(dat(1:4)),'-',num2str(dat(6:7)),'-',num2str(dat(9:10))];
title(dat1,'fontsize',18);
h=colorbar('v');
%caxis([-12 0]);

caxis([0 15]);
hold on
c = colorbar ('FontSize',18);
c.Label.String = 'depth (m)';
c.Label.FontSize = 18;

fig=strcat([sat,'_',dat,'_',region,'_SDB_Stumpf_bg.tif']); 
print(2,'-dtiff',fig,'-r600');

%%

figure(3)

SS = shaperead('vskp_watermask.shp');
isin = inpolygon(lon,lat,SS.X,SS.Y);

ZLm(isin) = NaN;


if strcmp(region,'Rushi_Beach')
    S = shaperead('vskp_shoreline-line.shp');
end

m_proj('Transverse Mercator','lon',LONLIMS,'lat',LATLIMS);
[XX,YY] = m_ll2xy(S.X,S.Y);
m_pcolor(lon,lat,ZLm)
shading 'interp';
hold on
geoshow(YY,XX,'Color','black');
colormap(flipud(parula));
%colormap(flipud(brewermap([],'*RdBu')));
m_grid('linewidth',2,'tickdir','out','tickstyle','dm','xtick',5,'ytick',5,'fontsize',18);
set(gca,'FontSize',30)
dat1 = [num2str(dat(1:4)),'-',num2str(dat(6:7)),'-',num2str(dat(9:10))];
title(dat1,'fontsize',18);
h=colorbar('v');
%caxis([-12 0]);

caxis([0 15]);
hold on
c = colorbar ('FontSize',18);
c.Label.String = 'depth (m)';
c.Label.FontSize = 18;

fig=strcat([sat,'_',dat,'_',region,'_SDB_Lyzenga.tif']); 
print(3,'-dtiff',fig,'-r600');



%% Validation with Testing data (independent check)
clear lt ln dep 

data = Testing;

dep = data(:,3);
dep = round(dep,3);
%dep = dep.*-1;
dep(dep==0)=0.01;

% data.d = dep;

lt = data(:,2);
ln = data(:,1);
dept = dep;
ln(dept<0)=NaN;
lt(dept<0)=NaN;
dept(dept<0)=NaN;

ln(dept>=20)=NaN;
lt(dept>=20)=NaN;
dept(dept>=20)=NaN;

lt(isnan(lt))='';
ln(isnan(ln))='';
dept(isnan(dept))=''; %dept=dept*-1;
[ss,pp] = size(dept);


clear LT LN DEP
LT = lt(lt>=LATLIMS(1) & lt <=LATLIMS(2) & ln>=LONLIMS(1) & ln<=LONLIMS(2));
LN = ln(lt>=LATLIMS(1) & lt <=LATLIMS(2) & ln>=LONLIMS(1) & ln<=LONLIMS(2));
DEPT  = dept(lt>=LATLIMS(1) & lt <=LATLIMS(2) & ln>=LONLIMS(1) & ln<=LONLIMS(2));
[ss,pp] = size(DEPT);

k=1;

win = 4.5e-4;
ln1 = zeros(ss,1); lt1= ln1; sdb1 = ln1; sdb2  = ln1;
disp('Please wait .. finding closest data')

tic
for k=1:ss

disp([num2str(k) ' out of ' num2str(ss)])
if LN(k)> min(min(lon)) && LN(k)< max(max(lon))
if LT(k)> min(min(lat)) && LT(k)< max(max(lat))
    
    sdbs = SDB_Stumpf_bg(lon<LN(k)+win & lon>LN(k)-win & lat<LT(k)+win & lat>LT(k)-win);
    sdbl = ZLm(lon<LN(k)+win & lon>LN(k)-win & lat<LT(k)+win & lat>LT(k)-win);
    lnn = lon(lon<LN(k)+win & lon>LN(k)-win & lat<LT(k)+win & lat>LT(k)-win);
    ltt = lat(lon<LN(k)+win & lon>LN(k)-win & lat<LT(k)+win & lat>LT(k)-win);
    dist = sqrt((LN(k)-lnn).^2+(LT(k)-ltt).^2);
    md = (min((dist)));
    [row,col] = find(dist==md);
    [si,sj] = size(col);
        if si >= 1 && sj>=1
            row1 = min(row);
            col1 = min(col);
            clear col;
            clear row;
            col = col1;
            row = row1;
        end
    ln1(k) = lnn(row,col);
    lt1(k) = ltt(row,col);
    sdb1(k) = sdbs(row,col);
    sdb2(k) = sdbl(row,col);
end
end
end
toc
%%

dept1 = DEPT;
sdb11 = sdb1;
ln1 = LN;
lt1 = LT;

dept2 = DEPT;
sdb22 = sdb2;
ln2 = LN;
lt2 = LT;

    dept1(isnan(sdb11))='';
    ln1(isnan(sdb11))='';
    lt1(isnan(sdb11))='';
    sdb11(isnan(sdb11))='';
    dept1(sdb11<=0)='';
    sdb11(sdb11<=0)='';
    
    P1 = polyfit(sdb11,dept1,1);
    
    yfit1 = P1(1)*sdb11+P1(2);
    disp(['m1 = ' num2str(P1(1)) ' and m0 = ' num2str(P1(2))])
    
    f1 = fit(sdb11,dept1,'poly1')
    %ci = confint(f1);
    
    
    dept2(isnan(sdb22))='';
    ln2(isnan(sdb22))='';
    lt2(isnan(sdb22))='';
    sdb22(isnan(sdb22))='';
    dept2(sdb22<=0)='';
    sdb22(sdb22<=0)='';
    P2 = polyfit(sdb22,dept2,1);
    
    yfit2 = P2(1)*sdb22+P2(2);
    disp(['m1 = ' num2str(P2(1)) ' and m0 = ' num2str(P2(2))])
    
    f2 = fit(sdb22,dept2,'poly1')
    %ci = confint(f);
    
  
%%

figure(4)

clear X

X(:,1) = dept1;
X(:,2) = sdb11;

diff1 = sdb11-dept1;

n = hist3(X,[100 100]); % default is to 10x10 bins
n1 = n';
n1(size(n,1) + 1, size(n,2) + 1) = 0;
xb = linspace(min(X(:,1)),max(X(:,1)),size(n,1)+1);
yb = linspace(min(X(:,2)),max(X(:,2)),size(n,1)+1);
n1(n1==0)=NaN;
h = pcolor(xb,yb,n1);
set(h,'EdgeColor','none');
box('on')
hold on

x=0:0.1:15;
y=0:0.1:15;
plot([x(1),x(end)],[y(1) y(end)],'-.k')
hold on
plot(f1,'--r')
legend('off')
colorbar
box on
xlim([0 15])
ylim([0 15])

xlabel('In-situ depth (m)','FontSize',18,'Fontweight','bold') 
ylabel('SDB LRM (m)','FontSize',18,'Fontweight','bold')
dat1 = [num2str(dat(1:4)),'-',num2str(dat(6:7)),'-',num2str(dat(9:10))];
title(dat1,'fontsize',20);
set(gca,'FontSize',20)
 
fig=strcat([sat,'_',dat,'_',region,'_Stumpf_validation.tif']); 
print(4,'-dtiff',fig,'-r600');


figure(5)
clear X

dept2(sdb22<=0)='';
sdb22(sdb22<=0)='';
X(:,1) = dept2;
X(:,2) = sdb22;

diff2 = sdb22-dept2;
n = hist3(X,[100 100]); % default is to 10x10 bins
n1 = n';
n1(size(n,1) + 1, size(n,2) + 1) = 0;
xb = linspace(min(X(:,1)),max(X(:,1)),size(n,1)+1);
yb = linspace(min(X(:,2)),max(X(:,2)),size(n,1)+1);
n1(n1==0)=NaN;
h = pcolor(xb,yb,n1);
set(h,'EdgeColor','none');
box('on')
hold on

x=0:0.1:15;
y=0:0.1:15;
plot([x(1),x(end)],[y(1) y(end)],'-.k')
hold on
plot(f1,'--r')
legend('off')
colorbar
box on
xlim([0 15])
ylim([0 15])

xlabel('In-situ depth (m)','FontSize',18,'Fontweight','bold') 
ylabel('SDB LLM (m)','FontSize',18,'Fontweight','bold')
dat1 = [num2str(dat(1:4)),'-',num2str(dat(6:7)),'-',num2str(dat(9:10))];
title(dat1,'fontsize',20);
set(gca,'FontSize',20)

fig=strcat([sat,'_',dat,'_',region,'_Lyzenga_validation.tif']); 
print(5,'-dtiff',fig,'-r600');

%%
figure(6)
histogram(diff1,'FaceColor',[0 0 0])
hold on
histogram(diff2,'FaceColor',[0.5 0.5 0.5])
legend('LRM','LLM')
xlim([-10 10])
xlabel('Residual error (m)','FontSize',18,'Fontweight','bold') 
ylabel('Pixel count','FontSize',18,'Fontweight','bold')
title(dat1,'fontsize',20);
set(gca,'FontSize',20)


fig=strcat([sat,'_',dat,'_',region,'_resi_error_hist.tif']); 
print(6,'-dtiff',fig,'-r600');


% %% Validation of entire points for Stumpf model

val1 = abs(sdb22); % Lyzenga
val3 = abs(dept2); % insitu

clc
disp('For Lyzenga...')
dd1 = (val1-val3);
dd3 = (val1-val3)./val3;
dd = (val1-val3).^2;
[mk,~]=size(dd);


% statistics computation
R = corr(val1,val3,'rows','complete');
RR2 = R*R
bias = sum(dd1)/mk % also known as Mean prediction difference (MPD)
RMSE = sqrt(sum(dd)/mk) % also known as Root mean sq pred diff (RMSPD)
%STDE = sqrt(sum((dd1 - ubias).^2)/mk)
dd2 = abs(dd1); 
MAE = sum(dd2)/mk
umsat = mean(val1);
umins = mean(val3);
udiff = umsat-umins;
uper = (abs(udiff)/umins)*100;
ust1 = std(val1);
ust2 = std(val3);
MNB = sum(dd3)/mk % Mean normalised bias
diffmed = median(val1)-median(val3)
mk

%%

val1 = abs(sdb11); % Stumpf
val2 = abs(dept1); % insitu

clc
disp('For Stumpf...')
dd1 = (val1-val2);
dd3 = (val1-val2)./val2;
dd = (val1-val2).^2;
[mk,~]=size(dd);

%clc
% statistics computation
R = corr(val1,val2,'rows','complete');
RR2 = R*R
bias = sum(dd1)/mk % also known as Mean prediction difference (MPD)
RMSE = sqrt(sum(dd)/mk) % also known as Root mean sq pred diff (RMSPD)

dd2 = abs(dd1); 
MAE = sum(dd2)/mk
umsat = mean(val1);
umins = mean(val2);
udiff = umsat-umins;
uper = (abs(udiff)/umins)*100;
ust1 = std(val1);
ust2 = std(val2);
MNB = sum(dd3)/mk % Mean normalised bias
diffmed = median(val1)-median(val2)
mk

%% Depth class-wise statistics
clc
sdb11= abs(sdb11);
sdb22= abs(sdb22);
dept1 = abs(dept1);
dept2 = abs(dept2);
diffa = sdb11-dept1;
diffb = sdb22-dept2;
for i = 1:6
switch i
    case 1
        
        sdep1 = sdb11(dept1>0 & dept1<=2);
        dep1 =dept1(dept1>0 & dept1<=2);
        diff1a = sdep1-dep1;
        
        sdep2 = sdb22(dept2>0 & dept2<=2);
        dep2 =dept2(dept2>0 & dept2<=2);
        diff1b = sdep2-dep2;
    case 2
        dep1 =dept1(dept1>2 & dept1<=4);
        sdep1 = sdb11(dept1>2 & dept1<=4);
        diff2a = sdep1-dep1;
        
        dep2 =dept2(dept2>2 & dept2<=4);
        sdep2 = sdb22(dept2>2 & dept2<=4);
        diff2b = sdep2-dep2;
    case 3
        dep1 =dept1(dept1>4 & dept1<=6);
        sdep1 = sdb11(dept1>4 & dept1<=6);
        diff3a = sdep1-dep1;
        
        dep2 =dept2(dept2>4 & dept2<=6);
        sdep2 = sdb22(dept2>4 & dept2<=6);
        diff3b = sdep2-dep2;
    case 4
        dep1 =dept1(dept1>6 & dept1<=8);
        sdep1 = sdb11(dept1>6 & dept1<=8);
        diff4a = sdep1-dep1;
        
        dep2 =dept2(dept2>6 & dept2<=8);
        sdep2 = sdb22(dept2>6 & dept2<=8);
        diff4b = sdep2-dep2;
    case 5
        dep1 =dept1(dept1>8 & dept1<=10);
        sdep1 = sdb11(dept1>8 & dept1<=10);
        diff5a = sdep1-dep1;
        
        dep2 =dept2(dept2>8 & dept2<=10);
        sdep2 = sdb22(dept2>8 & dept2<=10);
        diff5b = sdep2-dep2;
    case 6
        dep1 =dept1(dept1>10 & dept1<=12);
        sdep1 = sdb11(dept1>10 & dept1<=12);
        diff6a = sdep1-dep1;
        
        dep2 =dept2(dept2>10 & dept2<=12);
        sdep2 = sdb22(dept2>10 & dept2<=12);
        diff6b = sdep2-dep2;
 
end

disp('For Stumpf Model ....')
disp(['Statistics for case: ',num2str(i)])

val1 = sdep1;
val2 = dep1;
dd1 = (val1-val2);
dd3 = (val1-val2)./val2;
dd = (val1-val2).^2;
[mk,~]=size(dd);

mk
bias = sum(dd1)/mk % also known as Mean prediction difference (MPD)
RMSE = sqrt(sum(dd)/mk) % also known as Root mean sq pred diff (RMSPD)
R = corr(val1,val2,'rows','complete');
RR2 = R*R;
%STDE = sqrt(sum((dd1 - ubias).^2)/mk)
dd2 = abs(dd1); 
MAE = sum(dd2)/mk
MNB = sum(dd3)/mk % Mean normalised bias
umsat = mean(val1);
umins = mean(val2);
udiff = umsat-umins;
diffmed = median(val1)-median(val2)
uper = (abs(udiff)/umins)*100;
ust1 = std(val1);
ust2 = std(val2);

clear val1 val2 dd1 dd2 dd3 dd dd2 diffmed uper ust1
disp('For Lyzenga Model ....')
disp(['Statistics for case: ',num2str(i)])

val1 = sdep2;
val2 = dep2;
dd1 = (val1-val2);
dd3 = (val1-val2)./val2;
dd = (val1-val2).^2;
[mk,~]=size(dd);

mk
bias = sum(dd1)/mk % also known as Mean prediction difference (MPD)
RMSE = sqrt(sum(dd)/mk) % also known as Root mean sq pred diff (RMSPD)
R = corr(val1,val2,'rows','complete');
RR2 = R*R;
%STDE = sqrt(sum((dd1 - ubias).^2)/mk)
dd2 = abs(dd1); 
MAE = sum(dd2)/mk
MNB = sum(dd3)/mk % Mean normalised bias
umsat = mean(val1);
umins = mean(val2);
udiff = umsat-umins;
diffmed = median(val1)-median(val2)
uper = (abs(udiff)/umins)*100;
ust1 = std(val1);
ust2 = std(val2);
end

%% Histogram plots for depth classes for Stumpf model

p1 = diff1a;
p2 = diff2a;
p3 = diff3a;
p4 = diff4a;
p5 = diff5a;
p6 = diff6a;

binrange = -4:0.5:4;
hcp1 = histcounts(p1,[binrange Inf]); s = sum(hcp1); hcp1 = hcp1./s;
hcp2 = histcounts(p2,[binrange Inf]); s = sum(hcp2); hcp2 = hcp2./s;
hcp3 = histcounts(p3,[binrange Inf]); s = sum(hcp3); hcp3 = hcp3./s;
hcp4 = histcounts(p4,[binrange Inf]); s = sum(hcp4); hcp4 = hcp4./s;
hcp5 = histcounts(p5,[binrange Inf]); s = sum(hcp5); hcp5 = hcp5./s;
hcp6 = histcounts(p6,[binrange Inf]); s = sum(hcp6); hcp6 = hcp6./s;


figure (7)
bar(binrange,[hcp1;hcp2;hcp3;hcp4;hcp5;hcp6]')
lgd = legend('[Z=0-2 m]','[Z=2-4 m]','[Z=4-6 m]','[Z=6-8 m]','[Z=8-10 m]','[Z=10-12 m]','Location','northwest')
lgd.FontSize = 6;
xlabel('Differences between Z_{sat} and Z_{insitu}')
ylabel('Relative Frequency')
%lgd = legend();
title(lgd,'Depth class for Stumpf model')
 fig = gcf;
 fig.PaperUnits = 'inches';
 fig.PaperPosition = [0 0 6 3];
 
% print(gcf, '-r300','-dtiff', 'hist_tp.tiff');

%title('ROI 1 (S2B)','fontsize',14);
title('ROI 2 (S2B)','fontsize',14);
% fig=strcat([sat,'_',dat,'_',region,'_Stumpf_histo_dep_class.tif']); 
%     print(gcf,'-dtiff',fig,'-r600');


%%
%% Histogram plots for depth classes for Lyzenga model

p1 = diff1b;
p2 = diff2b;
p3 = diff3b;
p4 = diff4b;
p5 = diff5b;
p6 = diff6b;

binrange = -4:0.5:4;
hcp1 = histcounts(p1,[binrange Inf]); s = sum(hcp1); hcp1 = hcp1./s;
hcp2 = histcounts(p2,[binrange Inf]); s = sum(hcp2); hcp2 = hcp2./s;
hcp3 = histcounts(p3,[binrange Inf]); s = sum(hcp3); hcp3 = hcp3./s;
hcp4 = histcounts(p4,[binrange Inf]); s = sum(hcp4); hcp4 = hcp4./s;
hcp5 = histcounts(p5,[binrange Inf]); s = sum(hcp5); hcp5 = hcp5./s;
hcp6 = histcounts(p6,[binrange Inf]); s = sum(hcp6); hcp6 = hcp6./s;


figure (2)
bar(binrange,[hcp1;hcp2;hcp3;hcp4;hcp5;hcp6]')
lgd = legend('[Z=0-2 m]','[Z=2-4 m]','[Z=4-6 m]','[Z=6-8 m]','[Z=8-10 m]','[Z=10-12 m]','Location','northwest')
lgd.FontSize = 6;
xlabel('Differences between Z_{sat} and Z_{insitu}')
ylabel('Relative Frequency')
%lgd = legend();
title(lgd,'Depth class for Stumpf model')
 fig = gcf;
 fig.PaperUnits = 'inches';
 fig.PaperPosition = [0 0 6 3];
 
% print(gcf, '-r300','-dtiff', 'hist_tp.tiff');

%title('ROI 1 (S2B)','fontsize',14);
title('ROI 2 (S2B)','fontsize',14);
% fig=strcat([sat,'_',dat,'_',region,'_Stumpf_histo_dep_class.tif']); 
%     print(gcf,'-dtiff',fig,'-r600');


%% Depth difference plot for Stumpf model

if strcmp(region,'Rushi_Beach')
    
    S = shaperead('vskp_shoreline-line.shp');
end
if strcmp(region,'RK_Beach')
    S = shaperead('vskp_shoreline-line.shp');
end

if strcmp(region,'Calangute')
    S = shaperead('Goa_coast.shp');
end


if strcmp(region,'Neil')
    S = shaperead('Neil_shoreline.shp');
end

m_proj('Transverse Mercator','lon',LONLIMS,'lat',LATLIMS);
[X,Y] = m_ll2xy(ln1,lt1);
[XX,YY] = m_ll2xy(S.X,S.Y);

% figure (8)
% scatter(X,Y,10,diffa,'o','filled')
% hold on
% geoshow(YY,XX,'Color','black');
% m_grid('linewidth',2,'tickdir','out','tickstyle','dm','xtick',10,'ytick',10,'fontsize',12);
% box on
% colormap(brewermap([],'*RdBu'));
% c = colorbar;
% c.Label.String = 'depth difference (m)';
% c.Label.FontSize = 12;
% title('Depth difference (ROI 2) Stumpf','fontsize',16);
% caxis([-3 3])

% fig=strcat([region,'_Stumpf_diff_plot.tif']); 
% %fig=strcat([region,'_validation_data.tif']); 
% print(1,'-dtiff',fig,'-r600');
%%

