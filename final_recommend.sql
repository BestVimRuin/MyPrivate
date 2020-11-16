drop table aiservices.recommend_user_cluster_4;
create table aiservices.recommend_user_cluster_4 as
select client_id, curr_serv_mngr_no, curr_serv_dept_nozz, curr_serv_subc_nozz, 'ZZ001041' comId, 
       case when size(split(user_type,','))>3 then concat_ws(',', split(user_type,',')[0],split(user_type,',')[1],split(user_type,',')[2]) else user_type end user_type
from 
(select client_id, curr_serv_mngr_no, curr_serv_dept_nozz, curr_serv_subc_nozz, concat_ws(',',nlpf_type,jzkh_type,rjkh_type,xjlc_type,ylks_type,xlkh_type)  user_type
from 
(select client_id, curr_serv_mngr_no, curr_serv_dept_nozz, curr_serv_subc_nozz, 
       case when his_max_ast > 0 and is_professional=1 then   case when  client_age<40  then  case when  avg_ast_open_date<300000  then  '年轻大众客户'  else '年轻富裕客户' end
       	                                                           when  client_age>=40 and client_age<=60  then   case when  avg_ast_open_date<300000  then  '中年大众客户'   else '中年富裕客户' end  
       	                                                      else case when  avg_ast_open_date<300000  then  '老年大众客户'   else '老年富裕客户' end    end   end nlpf_type,
       case when tot_aset>1000000 and ((gp_mkt_val > 0 and gp_mkt_val > lc_hold_val) or stk_tran_cnt_m>=4 )        then '高净值交易客户'  end jzkh_type,
       case when bstr_in_amt_month>0 then case when lc_hold_val > 0 and lc_hold_val > gp_mkt_val then '理财入金客户'     when gp_mkt_val > 0  and gp_mkt_val > lc_hold_val then '交易入金客户'      end end rjkh_type,       
       case when asset>1000          then case when lc_hold_val > 0 and lc_hold_val > gp_mkt_val then '理财现金留存客户' when gp_mkt_val > 0  and gp_mkt_val > lc_hold_val then '交易现金留存客户'  end end xjlc_type,
       case when pl_val_year> 0	then case when rate_lc>0.8 then '理财盈利客户' when rate_gp>0.8 then '交易盈利客户' end when pl_val_year< 0 then case when rate_lc>0.8  then '理财亏损客户' when rate_gp>0.8 then '交易亏损客户' end end ylks_type,
       case when kh_days<365 then  '新客' end xlkh_type
from 
(select t1.brok_id client_id, t1.kh_days, t2.gp_mkt_val, t3.lc_hold_val, t2.tot_aset, t2.rate_gp, case when tot_aset is null or tot_aset=0 then 0 else lc_hold_val/tot_aset end rate_lc, t4.pl_val_year , t4.bstr_in_amt_month, t5.stk_tran_cnt_m, t6.asset, t7.client_age, t7.client_risk_grad, t7.client_ivsm_maty, t7.client_ivsm_var, t7.isn_prof_ivst, t7.is_professional, t7.curr_serv_mngr_no, t7.curr_serv_dept_nozz, t7.curr_serv_subc_nozz, t8.his_max_ast, t9.avg_ast_open_date 
from 
(select brok_id,pty_id, datediff(from_unixtime(unix_timestamp()), from_unixtime(unix_timestamp(open_dt,'yyyyMMdd'))) kh_days from src_easyetl.app_cust_info_c where hdfs_par = ana_fx_middle.get_closest_trade_day_in_history('${job.biz.date}')  and sys_id = '102010' ) t1
left join
(select ana_fx_middle.char_map1_encode(sor_pty_id) sor_pty_id, nvl(gp_mkt_val,0) gp_mkt_val, nvl(tot_aset,0) tot_aset, case when tot_aset is null or tot_aset=0 then 0 else nvl(gp_mkt_val,0)/tot_aset end rate_gp from src_ht_cda.app_cust_aset_info_mt where hdfs_par='$(date.preNDay("${job.biz.date}","dft",1))') t2
on t1.brok_id=t2.sor_pty_id
left join
(select client_id, sum(nvl(cast (hold_amt as decimal(38,4)),0))  lc_hold_val from  tg_htsc_dwa.pub_client_hold_dtl where hdfs_par = ana_fx_middle.get_closest_trade_day_in_history('${job.biz.date}') and period_id='$(date.preNDay("${job.biz.date}","dft",1))' group by client_id ) t3
on t1.brok_id=t3.client_id
left join
(select client_id, sum(nvl(pl_val_tday,0)) pl_val_year, sum(case when datediff(from_unixtime(unix_timestamp()), from_unixtime(unix_timestamp(hdfs_par,'yyyyMMdd')))<=30 then nvl(bstr_in_amt_tday,0) else 0 end ) bstr_in_amt_month from tg_htsc_dwa.pub_cust_tday_info_df  where hdfs_par>='$(date.preNDay("${job.biz.date}","dft",365))' group by client_id) t4
on t1.brok_id=t4.client_id
left join
(select pty_id, nvl(stk_tran_cnt_m,0) stk_tran_cnt_m from src_easyetl.app_cust_char_c  where hdfs_par = ana_fx_middle.get_closest_trade_day_in_history('${job.biz.date}') ) t5
on t1.pty_id=t5.pty_id
left join
(select client_id, nvl(cast(fund as  double),0) + nvl(cast(ttf_asset as double),0) asset from src_report.cft_client_push  where hdfs_par=ana_fx_middle.get_closest_trade_day_in_history('${job.biz.date}')) t6
on t1.brok_id=t6.client_id
left join
(select client_id,client_age,client_risk_grad,client_ivsm_maty, client_ivsm_var, isn_prof_ivst, case when (client_risk_grad is not null and client_ivsm_maty is not null and client_ivsm_var is not null) or isn_prof_ivst='Y' then 1 else 0 end is_professional, curr_serv_mngr_no, curr_serv_dept_nozz, curr_serv_subc_nozz  from tg_htsc_dwa.pub_cust_base_info_df where hdfs_par=ana_fx_middle.get_closest_trade_day_in_history('${job.biz.date}') ) t7
on t1.brok_id=t7.client_id 
left join
(select client_id, nvl(his_max_ast,0) his_max_ast  from tg_htsc_dwa.pub_cust_all_info_df  where hdfs_par=ana_fx_middle.get_closest_trade_day_in_history('${job.biz.date}')  ) t8
on t1.brok_id=t8.client_id
left join
(select client_id, nvl(avg_ast_open_date,0) avg_ast_open_date from tg_htsc_dwa.pub_cust_avg_info_df where hdfs_par='$(date.preNDay("${job.biz.date}","dft",1))') t9
on t1.brok_id=t9.client_id
) t
) tt) ttt;


drop table aiservices.recommend_user_buy_tmp;
create table aiservices.recommend_user_buy_tmp as
select client_id, fund_code , secucode, managementcomcode, total_tamt , total_tshare, total_num, level3_category
from 
(select client_id, fund_code , sum(nvl(cast(tamt as double),0)) total_tamt , sum(nvl(cast(tshare as double),0))  total_tshare, count(1) total_num
from tg_htsc_dwa.pub_client_entr_match 
where hdfs_par>='$(date.preNDay("${job.biz.date}","dft",365))'
and (busi_desc = '认购确认' and cast(mkt_id as double) = 9)    or -- 场外认购，busi_flag=120
    (busi_desc = '认购确认' and cast(mkt_id as double) = 109)  or -- 沪市场内，busi_flag=4077
    (busi_desc = '认购结果' and cast(mkt_id as double) = 109 ) or -- 沪市场内，busi_flag=4022
    (busi_desc = '认购结果' and cast(mkt_id as double) = 110)  or -- 深市场内，busi_flag=4022
    (busi_desc = '认购结果' and cast(mkt_id as double) > 1000) or -- OTC市场，busi_flag=120
    (busi_desc = '申购确认' and cast(mkt_id as double) = 9)    or -- 场外申购，busi_flag=122，139
    (busi_desc = '申购确认' and cast(mkt_id as double) > 1000) or -- OTC市场，busi_flag=44122
    (busi_desc = '基金买入' and cast(mkt_id as double) > 1000) or -- OTC市场，busi_flag=44711
    busi_desc = '报价回购'                                        -- 报价回购，busi_flag=4140
and nvl(cast(tamt as double),0)>0
group by client_id,fund_code
) t1
join
(select prdt_code, level3_category from src_center_admin.vw_prdt_info_analyse where hdfs_par='$(date.preNDay("${job.biz.date}","dft",1))') t2
on t1.fund_code=t2.prdt_code
join
(select secucode, tradingcode, managementcomcode  from src_center_admin.fnd_basicinfo where hdfs_par = substr('$(date.preNDay("${job.biz.date}","dft",1))',1,4) and hd_business_date='$(date.preNDay("${job.biz.date}","dft",1))')  t3
on t1.fund_code = t3.tradingcode;


drop table aiservices.recommend_user_category_1;
create table aiservices.recommend_user_category_1 as
select client_id, level3_category, sum(total_num) total_num from aiservices.recommend_user_buy_tmp group by client_id, level3_category;

drop table aiservices.recommend_user_managementcom_2;
create table aiservices.recommend_user_managementcom_2 as
select client_id, managementcomcode, sum(total_num) total_num from aiservices.recommend_user_buy_tmp group by client_id, managementcomcode;


drop table aiservices.recommend_user_manager_3;
create table aiservices.recommend_user_manager_3 as
select client_id, pcode, sum(total_num) total_num 
from 
(select client_id, secucode, total_num from aiservices.recommend_user_buy_tmp ) t1
join
(select secucode, pcode, pname from src_center_admin.fnd_manager  where hdfs_par = substr('$(date.preNDay("${job.biz.date}","dft",1))',1,4) and hd_business_date='$(date.preNDay("${job.biz.date}","dft",1))' and isposition='1' ) t2
on t1.secucode=t2.secucode
group by client_id, pcode




9	场外
109	上海
110	深圳
>1000	otc

prdt_id	产品id
prdt_code	产品代码
prdt_name	产品全称
prdt_abbr	产品简称
level1_category	一级分类（一级分类包括：公募基金、私募基金、紫金产品、收益凭证、报价回购）
level2_category	二级分类
level3_category	三级分类
level4_category	四级分类
large_asset_labels	大类资产标签（国内权益类、货币类……所有产品都有）
professional_check_labels	业务考核分类（仅针对公募基金，国内权益类、海外权益类、固定收益类、大宗商品类……所有产品都有）
dept_name	归属部门
risk_level	风险等级
prdt_inv_horizon	投资品种
prdt_inv_breed	投资期限
prod_owner	产品来源代码
prod_owner_name	产品来源（内部创设/外部引入）
public_raise_begin_date	募集起始日期
raise_end_date	募集结束日期
raise_begin_date	华泰销售起始日期
theme_labels	所属主题
fund_com	管理人（基金公司）
fund_manager	基金经理


select prdt_code, level3_category, fund_com, fund_manager from src_center_admin.vw_prdt_info_analyse where hdfs_par='$(date.preNDay("${job.biz.date}","dft",1))'

fund_code	理财产品代码
fund_company_name	基金公司

ID
PRDT_CODE
PRDT_NAME

select * from src_center_admin.prdt_basicinfo where  hdfs_par='ALL' 



TRADINGCODE	交易所交易代码（揭示净值代码）


select * from src_center_admin.fnd_basicinfo   where hdfs_par='2020' and hd_business_date='20201115' and tradingcode='260116'





MANAGEMENTCOMCODE	基金管理人编码
MANAGEMENTCOM	MANAGEMENTCOM
SECUABBR	证券简称
FUNDMANAGER	FUNDMANAGER
SECUCODE	证券编码
CHINAME	证券中文名称
ID	ID


