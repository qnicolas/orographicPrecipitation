import os
import glob
import time

import pandas as pd
import numpy as np
import xarray as xr

from scipy.ndimage import gaussian_filter
from scipy.stats import t,linregress,pearsonr

import sys
p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from orographicPrecipitation.precip_extremes_scaling import scaling
from orographicPrecipitation.betts_miller import sbm

# ==================================================================
# DISCRETIZATION PARAMETERS
# ==================================================================

timedisc=6
leveldisc= 1 #2
firstlevel= 8 #19 #400hpa ## 8 (150hpa)

CMIP6_FOLDER = "/global/cfs/projectdirs/m3522/cmip6/"

# ==================================================================
# BOXES AND RX5 EVENTS
# ==================================================================


box = {"nepal"     :{"small":np.array([80, 90, 25, 30]          )},
       "nepal2"    :{"small":np.array([80, 90, 25, 30]          )},
       "pakistan"  :{"small":np.array([62, 74, 24, 36]          )},
       "ghats"     :{"small":np.array([74, 77, 11, 15]          )},
       "vietnam"   :{"small":np.array([107, 110, 11, 16]        )},
       "mexico"    :{"small":np.array([360-110,360-100, 18, 30] )},
       "chile"     :{"small":np.array([360-72, 360-68, -37, -25])},
       "Candes"    :{"small":np.array([360-72,360-62,-18,-12]   )},
       "Sandes"    :{"small":np.array([360-74,360-70,-45,-33]   )},
       "california":{"small":np.array([360-124,360-120,37,43]   )},
       "cascades"  :{"small":np.array([360-124,360-121,46,49]   )},
      }

for k in box.keys():
    box[k]["medium"] = box[k]["small"]+ 15*np.array([-1,1,-1,1])
    box[k]["large"]  = box[k]["small"]+ 30*np.array([-1,1,-1,1])

rx5 = {"nepal"     : ["19980706","19990710","20000608","20011003","20020721","20030710","20040708","20050711","20060708","20070724","20080615","20090727","20100719","20110718","20120915","20130627"],
       "nepal2"     : ["19980706","19990710","20000608","20011003","20020721","20030710","20040708","20050711","20060708","20070724","20080615","20090727","20100719","20110718","20120915","20130627"],
       "pakistan"  : ["20000714","20010711","20020227","20030217","20041229","20050210","20061204","20070627","20080803","20090723","20100727","20110831","20120907","20130203"],
       "ghats"     :["19980717","19990719","20000805","20010801","20021011","20030621","20040802","20050714","20060713","20070621","20080812","20090705","20100729","20110718","20120808","20130703"],
       "vietnam"   :["19981113","19991103","20001008","20011021","20020816","20031016","20040610","20051030","20060922","20070803","20081116","20090927","20101103","20111106","20121115","20130916"],
       "mexico"    : ["20000915","20010910","20020917","20030921","20040902","20051008","20060915","20070903","20080823","20091019","20100703","20110825","20120927","20130917"],
       "chile"     : ["20000626","20010730","20020515","20030522","20040723","20050628","20060712","20070328","20080520","20090814","20100616","20110826","20120526","20130529"],
       "Candes"    : ["19980220","19990318","20000313","20010218","20020222","20030315","20040113","20050114","20060113","20070215","20080131","20090213","20100213","20110220","20120215","20131219"],
       "Sandes"    : ["19980617","19990619","20000602","20010526","20020524","20030620","20040615","20050628","20060606","20070905","20080520","20090812","20100622","20110608","20120613","20130529"],
       "california": ["19980116","19990214","20000112","20011230","20021215","20031130","20041208","20051229","20060101","20070209","20080104","20090303","20101227","20110316","20121130","20130402"],
       "cascades"  : ["19981123","19990115","20001018","20011215","20020107","20031018","20040823","20051102","20061105","20071204","20081111","20090106","20101210","20110114","20121121","20130405"],
      }


p99_trmm = {"nepal"     : ['19980620', '19980705', '19980706', '19980727', '19980802', '19980830', '19980831', '19981018', '19990709', '19990710', '19990822', '19990824', '19990925', '20000608', '20000621', '20000705', '20000706', '20000907', '20010901', '20011004', '20020720', '20020811', '20030626', '20030708', '20030709', '20030730', '20030817', '20030818', '20040706', '20040707', '20040708', '20050710', '20060708', '20060924', '20070726', '20070811', '20080613', '20080614', '20080627', '20080919', '20090727', '20090805', '20091006', '20091007', '20100710', '20100718', '20100719', '20100911', '20110630', '20110719', '20110805', '20110822', '20120705', '20120719', '20120724', '20120913', '20120917', '20130617', '20131013'],
       "ghats"     : ['19980601', '19980623', '19980624', '19980718', '19980719', '19980728', '19980925', '19990612', '19990715', '19990716', '19990717', '19990721', '19990722', '20000606', '20000612', '20000619', '20000711', '20000712', '20000803', '20000804', '20000805', '20010611', '20010803', '20021012', '20021013', '20030620', '20030621', '20030622', '20030715', '20040609', '20040708', '20040803', '20040804', '20050716', '20060629', '20060712', '20060713', '20060811', '20070622', '20070805', '20070909', '20080810', '20080813', '20090605', '20090704', '20090705', '20090715', '20090717', '20100705', '20100729', '20110602', '20110716', '20110802', '20120617', '20130611', '20130617', '20130618', '20130703', '20130719'],
       "vietnam"   : ['19981001', '19981002', '19981003', '19981021', '19981112', '19981113', '19981119', '19981125', '19981213', '19991023', '19991024', '19991101', '19991102', '19991103', '19991104', '19991105', '19991210', '19991211', '19991215', '19991216', '20000531', '20000820', '20000821', '20001008', '20001009', '20001116', '20011020', '20011021', '20011111', '20020818', '20021025', '20031015', '20031016', '20031112', '20040612', '20050912', '20051031', '20051101', '20070804', '20071001', '20071002', '20071103', '20071123', '20081117', '20090928', '20090929', '20091016', '20091017', '20091102', '20100823', '20101103', '20101108', '20111106', '20111107', '20121114', '20130917', '20131014', '20131106', '20131115'],
       "pakistan"  : [],
       "mexico"    : [],
       "chile"     : [],
       "Candes"    : ['19980113', '19980219', '19980928', '19981014', '19981111', '19981130', '19981222', '19990105', '19990319', '19991103', '19991120', '20000315', '20001212', '20010104', '20010217', '20010227', '20010524', '20011125', '20020106', '20020115', '20020206', '20030119', '20030214', '20030317', '20031228', '20040105', '20040115', '20040214', '20040504', '20050116', '20050126', '20050215', '20051207', '20060111', '20060214', '20070121', '20070215', '20070216', '20070413', '20080110', '20080202', '20080302', '20081211', '20081223', '20090215', '20090424', '20091204', '20110214', '20110218', '20111122', '20111225', '20120125', '20120210', '20120323', '20130126', '20130203', '20130413', '20131217', '20131219'],
       "Sandes"    : ['19980527', '19990619', '19990620', '19990621', '19990728', '19990912', '20000208', '20000602', '20000603', '20000612', '20000627', '20000909', '20001011', '20010130', '20010526', '20010527', '20010729', '20010828', '20020225', '20020314', '20020525', '20020824', '20021012', '20030614', '20030619', '20030707', '20030817', '20030904', '20040411', '20040613', '20040701', '20050528', '20050626', '20050627', '20060418', '20060524', '20060607', '20060711', '20080518', '20080519', '20080520', '20080522', '20080603', '20080815', '20090814', '20100623', '20110421', '20110519', '20110607', '20110618', '20110809', '20120526', '20120601', '20120612', '20120815', '20121224', '20130527', '20130627', '20130702'],
       "california": ['19980114', '19980116', '19980117', '19980118', '19980126', '19980203', '19980206', '19980212', '19980214', '19980216', '19980219', '19980322', '19981024', '19981121', '19981122', '19981224', '19990206', '19990212', '19990216', '19990319', '19991129', '20000111', '20000123', '20000213', '20000222', '20000223', '20010304', '20011112', '20011214', '20020106', '20021108', '20021214', '20021219', '20030216', '20030315', '20031130', '20031229', '20040216', '20040217', '20041019', '20041026', '20041208', '20051222', '20051228', '20051231', '20060227', '20061226', '20080104', '20081101', '20090301', '20091013', '20101024', '20101107', '20101219', '20101229', '20120121', '20121130', '20121202', '20121223'],
       "cascades"  : ['19980114', '19980309', '19980323', '19981012', '19981120', '19981121', '19981125', '19981212', '19981224', '19981225', '19981228', '19990114', '19990117', '19990129', '19990202', '19990212', '19990227', '19990312', '19991027', '19991106', '19991109', '19991209', '19991215', '20000116', '20000201', '20010202', '20011114', '20011128', '20011213', '20020107', '20020311', '20030131', '20030313', '20031020', '20031128', '20050117', '20050326', '20051031', '20060110', '20061106', '20061107', '20061110', '20070102', '20070324', '20071203', '20081112', '20090107', '20090108', '20091017', '20091119', '20100111', '20101024', '20101107', '20101208', '20101212', '20110116', '20111122', '20121119', '20121225'],
      }

p99_era5 = {"nepal"     : ['19790723', '19790724', '19800608', '19800719', '19810628', '19810727', '19830702', '19830823', '19831011', '19840618', '19840708', '19840723', '19840724', '19840906', '19850724', '19851017', '19870724', '19870810', '19870811', '19870812', '19871019', '19880705', '19880706', '19880731', '19880812', '19880825', '19890527', '19890712', '19890714', '19890728', '19900708', '19900709', '19900727', '19900812', '19900813', '19910803', '19940815', '19950618', '19950812', '19950813', '19950814', '19951110', '19960627', '19960711', '19960712', '19960713', '19960813', '19961004', '19970629', '19970708', '19970709', '19970710', '19970711', '19970811', '19971209', '19980707', '19980708', '19980718', '19980813', '19980818', '19990612', '19990627', '19990628', '19990702', '19990710', '19990711', '19990825', '19991019', '20000607', '20000608', '20000622', '20000731', '20000801', '20010715', '20010729', '20020720', '20020721', '20030708', '20030709', '20030730', '20030818', '20030819', '20031009', '20040618', '20040706', '20040707', '20040708', '20041007', '20050623', '20050716', '20060708', '20070717', '20070725', '20070726', '20070727', '20070814', '20070905', '20070925', '20080614', '20080722', '20080919', '20090727', '20090728', '20090806', '20090817', '20090818', '20091006', '20091007', '20100710', '20100711', '20100718', '20100719', '20100818', '20100821', '20100822', '20100823', '20100824', '20110629', '20110630', '20110816', '20110817', '20120705', '20120706', '20120715', '20120724', '20120914', '20120916', '20120917', '20130617', '20130628', '20131014', '20140813', '20140814', '20140815', '20141014', '20160701', '20160717', '20160722', '20170704', '20170705', '20170709', '20170710', '20170810', '20170811', '20170812', '20180701', '20180702'],
       "ghats"     : ['19790617', '19800603', '19800701', '19800702', '19810605', '19810731', '19820616', '19820728', '19820729', '19820730', '19820731', '19820801', '19820802', '19830712', '19830713', '19830714', '19830715', '19830720', '19830721', '19830801', '19830808', '19830811', '19840617', '19840618', '19840630', '19840713', '19840714', '19850624', '19860804', '19870605', '19880717', '19880817', '19890608', '19890716', '19890717', '19890722', '19890723', '19900611', '19900707', '19900809', '19900810', '19910604', '19910605', '19910606', '19910607', '19910626', '19910707', '19910708', '19910709', '19920716', '19940602', '19940714', '19950612', '19950613', '19950710', '19950711', '19960614', '19960615', '19960616', '19960617', '19960618', '19960619', '19960714', '19960715', '19960716', '19960717', '19970630', '19970701', '19970724', '19970725', '19980625', '19980627', '19980628', '19980629', '19990722', '20000606', '20000607', '20000824', '20000825', '20010611', '20021012', '20030619', '20030620', '20030621', '20030705', '20030715', '20040504', '20040608', '20040609', '20050621', '20050904', '20060527', '20060528', '20060529', '20060530', '20060531', '20060713', '20070618', '20070619', '20070622', '20070623', '20070802', '20080727', '20080810', '20080813', '20081024', '20090606', '20090630', '20090702', '20090703', '20090715', '20090717', '20091001', '20091002', '20091110', '20100612', '20100613', '20100614', '20100730', '20100816', '20110602', '20110715', '20110716', '20110803', '20110820', '20110831', '20110901', '20120617', '20120618', '20120811', '20120829', '20120830', '20130531', '20130602', '20130618', '20130619', '20140508', '20140713', '20160622', '20160628', '20160704', '20170610', '20170611', '20180527', '20180529', '20180608', '20180814'],
       "vietnam"   : ['19790621', '19790622', '19790623', '19791014', '19800828', '19800910', '19801026', '19801101', '19801102', '19801116', '19811014', '19811108', '19811109', '19811110', '19811113', '19811201', '19820627', '19820906', '19830624', '19830625', '19831008', '19831029', '19840609', '19841012', '19841101', '19841107', '19850617', '19850618', '19851125', '19860518', '19861021', '19861201', '19861202', '19870815', '19870821', '19871118', '19880921', '19881009', '19881015', '19881106', '19890524', '19890722', '19900615', '19901002', '19901013', '19901014', '19901018', '19901111', '19901112', '19901115', '19910316', '19921022', '19921023', '19921028', '19931002', '19931003', '19931123', '19931124', '19951006', '19951026', '19951031', '19951101', '19960517', '19960911', '19960912', '19961102', '19961219', '19970921', '19971102', '19980930', '19981112', '19981113', '19981119', '19981125', '19981210', '19981213', '19981214', '19990428', '19991017', '19991018', '19991103', '19991104', '19991105', '19991202', '19991215', '20000531', '20001007', '20001010', '20001017', '20011021', '20011111', '20031112', '20040612', '20050912', '20051214', '20051220', '20060701', '20060702', '20060930', '20061001', '20061107', '20070804', '20070805', '20071001', '20071025', '20071103', '20071104', '20071122', '20071123', '20080124', '20080512', '20080513', '20081117', '20090905', '20090906', '20090907', '20090908', '20090928', '20090929', '20091102', '20100725', '20101030', '20101103', '20101109', '20101110', '20110922', '20110923', '20110924', '20111106', '20111107', '20120401', '20121006', '20130917', '20130918', '20131014', '20131106', '20141212', '20141216', '20160626', '20160912', '20161214', '20161215', '20171103', '20171104', '20180602', '20180603', '20181124'],
       "pakistan"  : [],
       "mexico"    : [],
       "chile"     : [],
       "Candes"    : ['19791215', '19791222', '19791231', '19800112', '19800208', '19800213', '19801109', '19801130', '19801220', '19810114', '19810215', '19810317', '19811119', '19820122', '19820327', '19821104', '19821207', '19821223', '19830127', '19830201', '19830213', '19830310', '19831023', '19831122', '19831231', '19850127', '19850801', '19851112', '19860105', '19860118', '19860119', '19860201', '19860525', '19870201', '19870310', '19871205', '19871218', '19880106', '19880107', '19880312', '19881122', '19881201', '19890224', '19891212', '19891213', '19891218', '19900111', '19900118', '19900226', '19900327', '19900418', '19920216', '19920322', '19921104', '19930101', '19930109', '19930214', '19940109', '19940301', '19941029', '19941118', '19941221', '19941223', '19950112', '19950113', '19951118', '19960120', '19970102', '19970120', '19970206', '19971114', '19980211', '19980219', '19980929', '19981126', '19981210', '19981222', '19991120', '19991206', '20000214', '20001115', '20011130', '20011223', '20030124', '20031011', '20031229', '20040114', '20040115', '20040205', '20050103', '20050116', '20050126', '20050314', '20051125', '20051207', '20060111', '20060210', '20060214', '20061016', '20061107', '20070215', '20070216', '20070427', '20080211', '20081226', '20090215', '20100121', '20101221', '20110123', '20110206', '20110210', '20110214', '20110219', '20111122', '20111225', '20120113', '20120216', '20121227', '20130102', '20130126', '20130203', '20130326', '20131219', '20140104', '20140125', '20140126', '20140129', '20140214', '20141113', '20150207', '20150217', '20150220', '20150221', '20151206', '20160114', '20160131', '20160201', '20161228', '20170125', '20170411', '20171119', '20171205', '20171225', '20180222', '20181124', '20181202', '20181224'],
       "Sandes"    : ['19790726', '19790728', '19790824', '19790830', '19800211', '19800410', '19800510', '19800615', '19800627', '19810502', '19810504', '19810507', '19810523', '19820510', '19820512', '19820715', '19820716', '19820824', '19820912', '19820913', '19830616', '19830617', '19830618', '19840502', '19840609', '19840701', '19840704', '19840716', '19850524', '19850702', '19860420', '19860515', '19860526', '19860527', '19860612', '19860613', '19860615', '19860616', '19860821', '19861125', '19870506', '19870608', '19870709', '19870711', '19870714', '19870724', '19870811', '19880619', '19880628', '19880728', '19890628', '19890726', '19900328', '19900408', '19910526', '19910527', '19910528', '19910708', '19920604', '19920605', '19920916', '19930503', '19930603', '19930625', '19931201', '19940426', '19940718', '19940719', '19940720', '19940723', '19950429', '19950622', '19950623', '19970422', '19970423', '19970603', '19970619', '19970620', '20000602', '20000603', '20000604', '20000612', '20000613', '20000627', '20000630', '20000909', '20010130', '20010521', '20010526', '20010527', '20010528', '20010630', '20010703', '20010719', '20010828', '20020525', '20020526', '20020720', '20020823', '20020824', '20021012', '20030614', '20030619', '20030707', '20040609', '20040701', '20040712', '20050528', '20050626', '20050627', '20060419', '20060525', '20060607', '20060711', '20060810', '20080518', '20080519', '20080520', '20080522', '20080712', '20080815', '20090516', '20090814', '20090905', '20100622', '20100623', '20110714', '20120526', '20120527', '20120528', '20120601', '20120612', '20130527', '20130627', '20130702', '20130804', '20140608', '20140611', '20140728', '20150602', '20150605', '20150708', '20150828', '20170616', '20170622', '20170813', '20171004'],
       "california": ['19790111', '19790213', '19791025', '19791224', '19800112', '19800113', '19800218', '19800219', '19801202', '19801203', '19810122', '19810127', '19811028', '19811116', '19811219', '19811220', '19820104', '19820215', '19820301', '19820331', '19820411', '19821118', '19821221', '19821222', '19830124', '19830126', '19830127', '19830301', '19830313', '19831110', '19831124', '19831211', '19831224', '19831225', '19841102', '19850208', '19851021', '19851202', '19860116', '19860214', '19860215', '19860216', '19860217', '19860218', '19860219', '19860308', '19870213', '19870305', '19871202', '19871206', '19881122', '19881123', '19890302', '19890309', '19891023', '19900107', '19900527', '19910303', '19910304', '19911026', '19921209', '19930120', '19931208', '19940217', '19950108', '19950109', '19950110', '19950114', '19950309', '19950310', '19951211', '19951212', '19951215', '19960116', '19960127', '19960204', '19960219', '19960516', '19961118', '19961205', '19961210', '19961231', '19970101', '19970102', '19970125', '19970126', '19971126', '19980203', '19980206', '19980221', '19981121', '19981123', '19981130', '19990207', '19990209', '20000111', '20000124', '20000214', '20000227', '20021108', '20021214', '20021216', '20021228', '20030315', '20031229', '20040101', '20040216', '20040217', '20040225', '20041019', '20041208', '20051201', '20051218', '20051222', '20051228', '20051230', '20051231', '20060227', '20060228', '20080104', '20091013', '20100118', '20100120', '20101024', '20101219', '20101229', '20110216', '20120121', '20121130', '20121202', '20141211', '20150206', '20150207', '20151213', '20160306', '20160313', '20161014', '20161210', '20161215', '20170104', '20170108', '20170110', '20170207', '20170209', '20170220', '20171116', '20180322'],
       "cascades"  : ['19791214', '19791217', '19800112', '19801107', '19801121', '19801225', '19801226', '19810216', '19811006', '19811205', '19820116', '19820123', '19820213', '19820214', '19821203', '19821216', '19830105', '19830329', '19831103', '19831115', '19841102', '19841127', '19850607', '19860118', '19861026', '19861120', '19861123', '19861124', '19870201', '19870303', '19871209', '19880114', '19881230', '19891109', '19891204', '19900107', '19900109', '19900210', '19901109', '19901110', '19901122', '19901124', '19910219', '19910404', '19910405', '19930125', '19931210', '19941031', '19941130', '19941217', '19941220', '19941226', '19941227', '19950131', '19950219', '19951010', '19951108', '19951111', '19951129', '19960206', '19960208', '19960423', '19961127', '19961229', '19970101', '19970319', '19970917', '19971030', '19971216', '19981113', '19981120', '19981121', '19981125', '19981202', '19981229', '19990114', '19990202', '19990224', '19991112', '19991202', '19991215', '20000201', '20000612', '20011114', '20011213', '20011216', '20020107', '20020125', '20020311', '20021214', '20030131', '20030322', '20031016', '20031017', '20031020', '20031118', '20031119', '20041210', '20050118', '20051031', '20060110', '20060130', '20061106', '20061107', '20070102', '20071203', '20081107', '20081112', '20090107', '20090108', '20091017', '20091026', '20091116', '20091117', '20091119', '20101212', '20110116', '20111122', '20111123', '20111228', '20121119', '20121217', '20130928', '20130929', '20140111', '20140217', '20141022', '20150105', '20150118', '20150206', '20151031', '20151113', '20151117', '20151207', '20151209', '20160310', '20161014', '20170118', '20170209', '20171019', '20171022', '20171120', '20171219', '20171229', '20181028', '20181127', '20181218'],
      }


rx5_datetime = {k:[pd.to_datetime(d, format='%Y%m%d') for d in rx5[k]] for k in rx5}
p99_trmm_datetime = {k:[pd.to_datetime(d, format='%Y%m%d') for d in p99_trmm[k]] for k in p99_trmm}
p99_era5_datetime = {k:[pd.to_datetime(d, format='%Y%m%d') for d in p99_era5[k]] for k in p99_era5}

names = {"nepal"     :"Nepal"     ,
         "nepal2"    :"Nepal - sanity check (calculations done a second time)"     ,
         "pakistan"  :"Pakistan"  ,
         "ghats"     :"Central Western Ghats"  ,
         "vietnam"   :"Annamite range (Viet-Nam)"  ,
         "mexico"    :"Mexico"    ,
         "chile"     :"Chile"     ,
         "Candes"    :"Central Andes",
         "Sandes"    :"South Andes",
         "california":"California",
         "cascades"  :"Cascade range",
      }


orog1 = xr.open_dataset(CMIP6_FOLDER+"ERA5/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc")
orog=orog1.Z/9.80665

m_per_degreelat = 6370*1e3*np.pi/180

coslat = np.cos(orog.latitude*np.pi/180.)
coslat += 1e-5*(1-1*(coslat>1e-5))

ddxorog = orog.differentiate("longitude")/(m_per_degreelat*coslat)
ddyorog = orog.differentiate("latitude")/m_per_degreelat
#Gaussian smoothing
ddxorog_s = xr.apply_ufunc(gaussian_filter,ddxorog,kwargs={"sigma":3.})
ddyorog_s = xr.apply_ufunc(gaussian_filter,ddyorog,kwargs={"sigma":3.})


# ==================================================================
# FUNCTIONS TO GATHER DATA FROM ERA5
# ==================================================================

def retrieve_era5_pl(ds,lonlat,varid,firstlev=firstlevel,levdisc=leveldisc,tdisc=timedisc):
    """gather a 'pressure levels' (i.e. 3D in space) ERA5 variable for days in ds, inside lonlat box,
    with a time discretization of "timedisc" hours (timedisc being defined above)
    and levels in range(firstlev,37,levdisc)
    varid gives the id of the variable in era5, e.g. '128_135_w' for omega
    """
    if type(ds)!=list:
        raise TypeError("first argument must be a list of days")
        
    era5vars = []
    for d in ds:
        era5var = xr.open_dataset(glob.glob(CMIP6_FOLDER+"ERA5/e5.oper.an.pl/*/e5.oper.an.pl.%s.*.%s00_%s23.nc"%(varid,d,d))[0])
        varname = list(era5var.data_vars)[0] #get name of the main variable, eg 'W' for omega
        era5var1 = era5var[varname].sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).isel(time=range(0,24,tdisc),level=range(firstlev,37,levdisc))
        era5vars.append(era5var1)
    return xr.concat(era5vars,'time')

def retrieve_era5_sfc(ds,lonlat,varid,tdisc=timedisc):
    """gather a surface era5 variable for days in ds, inside lonlat box
    with a time discretization of "timedisc" hours (timedisc being defined above)
    varid gives the id of the variable in era5, e.g. '128_134_sp' for surface pressure """
    if type(ds)!=list:
        raise TypeError("first argument must be a list of days")
    pslist=[]
    for d in ds:
        monthbegin = d[:-2]+"01"
        p = xr.open_dataset(glob.glob(CMIP6_FOLDER+"ERA5/e5.oper.an.sfc/*/e5.oper.an.sfc.%s.*.%s00_*.nc"%(varid,monthbegin))[0])
        varname = list(p.data_vars)[0] #get name of the main variable, eg 'PS' for surface pressure
        p1 = p[varname].sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]),time=d).isel(time=range(0,24,tdisc))
        pslist.append(p1)
    return xr.concat(pslist,'time')


# ==================================================================
# FUNCTIONS TO GATHER DATA FROM TRMM/GPCP
# ==================================================================


def precipTRMM1d(d,lonlat):
    """compute 3-hourly precip from TRMM in a box given by lonlat, on day d"""
    dp = pd.to_datetime(d, format='%Y%m%d')
    d2 = dp.strftime("%Y-%m-%d")
    #load monthly data
    pr = xr.open_dataset(glob.glob(CMIP6_FOLDER+"obs4mip/NASA-GSFC/TRMM/observations/atmos/pr/3hr/NASA-GSFC/TRMM/*/*%s*.nc"%d[:-2])[0])
    p = pr.sel(time = d2,lon=slice(lonlat[0],lonlat[1]),lat=slice(lonlat[2],lonlat[3]))
    
    return p.pr.mean("time")*24*3600

def precipTRMM5d(d,lonlat):
    """compute total precip from TRMM in a box given by lonlat, five days around d"""
    dp = pd.to_datetime(d, format='%Y%m%d')
    ds = [(dp + pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range(-2,3)]

    prs=[]
    for d in ds:
        prs.append(precipTRMM1d(d,lonlat))
    pr = xr.concat(prs,'time')
    
    sumpr = pr.mean("time")*5
    
    return sumpr

def precipGPCP1d(d,lonlat):
    """compute total precip from GPCP in a box given by lonlat, on day d"""
    gpcp = xr.open_dataset(CMIP6_FOLDER+"obs4mip/observations/NASA-GSFC/Obs-GPCP/GPCP/1DD_v1.3/pr_GPCP-1DD_L3_v1.3_19961001-20161231.nc")
    pr1d = gpcp.pr.sel(time=d,lon=slice(lonlat[0],lonlat[1]),lat=slice(lonlat[2],lonlat[3]))
    sumpr = pr1d.mean("time")*24*3600
    
    return sumpr

def precipGPCP5d(d,lonlat):
    """compute total precip from GPCP in a box given by lonlat, five days around d"""
    dp = pd.to_datetime(d, format='%Y%m%d')
    d1 = (dp + pd.Timedelta(days=-2)).strftime("%Y-%m-%d")
    d2 = (dp + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
   
    gpcp = xr.open_dataset(CMIP6_FOLDER+"obs4mip/observations/NASA-GSFC/Obs-GPCP/GPCP/1DD_v1.3/pr_GPCP-1DD_L3_v1.3_19961001-20161231.nc")
    pr5d = gpcp.pr.sel(time=slice(d1,d2),lon=slice(lonlat[0],lonlat[1]),lat=slice(lonlat[2],lonlat[3]))
    
    sumpr = pr5d.mean("time")*5*24*3600
    
    return sumpr

def precipERA51d(d,lonlat):
    """compute total precip from ERA5 in a box given by lonlat, on day d"""
    era5 = xr.open_dataset(CMIP6_FOLDER+"ERA5/e5.generated_tp/ERA5_IVT_tp_reanalysis_%s.nc"%d[:4])
    pr1d = era5.tp.sel(time=d,longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]))
    sumpr = pr1d.mean("time")*1000*24 #convert m to mm
    
    return sumpr

def precipERA55d(d,lonlat):
    """compute total precip from ERA5 in a box given by lonlat, five days around d"""
    dp = pd.to_datetime(d, format='%Y%m%d')
    d1 = (dp + pd.Timedelta(days=-2)).strftime("%Y-%m-%d")
    d2 = (dp + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
   
    era5 = xr.open_dataset(CMIP6_FOLDER+"ERA5/e5.generated_tp/ERA5_IVT_tp_reanalysis_%s.nc"%d[:4])
    pr5d = era5.tp.sel(time=slice(d1,d2),longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]))
    
    sumpr = pr5d.mean("time")*1000*5*24 #convert m to mm
    
    return sumpr





# ==================================================================
# FUNCTIONS TO COMPUTE SPATIAL MEANS ON ARRAYS
# ==================================================================

def spacemean_era5(ds,lonlat):
    return ds.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).mean(["latitude","longitude"])

def spacemean_trmm(ds,lonlat):
    return ds.sel(lon=slice(lonlat[0],lonlat[1]),lat=slice(lonlat[2],lonlat[3])).mean(["lat","lon"])


# ==================================================================
# FUNCTIONS TO BE VECTORIZED BY XARRAY (linear regression / correlations)
# ==================================================================

def lr(a,b):
    l,_,_,p,_=linregress(a,b)
    return l
def lrp(a,b):
    l,_,_,p,_=linregress(a,b)
    return p   
def corrc(a,b):
    c,p=pearsonr(a,b)
    return c
def corrp(a,b):
    c,p=pearsonr(a,b)
    return p   

def rsquared(x,y):
    """Coefficient of determination R^2 to measure fit of x to observed data y"""
    ybar = np.mean(y)
    return 1 - np.sum((y-x)**2)/np.sum((y-ybar)**2)


# ==================================================================
# UPSLOPE FLOW COMPUTATION
# ==================================================================

def upslope_w(ds,lonlat,firstlev=firstlevel,levdisc=leveldisc):
    """compute the upslope vertical velocity (w = U.grad(orography)) for days in ds, inside lonlat box, 
    with a time discretization of "timedisc" hours (timedisc being defined above) and levels in 
    range(levelbegin,37,leveldisc). 
    
    Args:
        ds : list of days for which the calculation should be processed; format "YYYYMMDD" (string)
        lonlat : list, [lon1, lon2, lat1, lat2] specifying the box on which to perform calculation. 
            NOTE THAT 0 <= lon1 < lon2 <= 360 and -90 <= lat1 < lat2 <= 90
        firstlev : int, specifies the index of the highest level to be used, as specified by the ERA5 
            reference pressure levels. E.G. 8 -> 150 hPa
        levdisc : int, specifies the level discretization to be used (relative to ERA5 reference 
        pressure levels)

    Returns:
        w : vertical velocity; 4D xarray with dimensions latitude, longitude, level, time
    """
    if type(ds)!=list:
        raise TypeError("first argument must be a list of days")
    ws = []
    
    ddx1=ddxorog_s.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).isel(time=0)
    ddy1=ddyorog_s.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).isel(time=0)

    for d in ds:
        u = xr.open_dataset(glob.glob(CMIP6_FOLDER+"ERA5/e5.oper.an.pl/*/e5.oper.an.pl.128_131_u.*.%s00_%s23.nc"%(d,d))[0]).U
        v = xr.open_dataset(glob.glob(CMIP6_FOLDER+"ERA5/e5.oper.an.pl/*/e5.oper.an.pl.128_132_v.*.%s00_%s23.nc"%(d,d))[0]).V
        u1 = u.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).isel(time=range(0,24,timedisc),level=range(firstlev,37,levdisc))
        v1 = v.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).isel(time=range(0,24,timedisc),level=range(firstlev,37,levdisc))
        ws.append(u1*ddx1+v1*ddy1)
        
    w=xr.concat(ws,'time')
    return w

def upslope_omega_mean200(ps,wupslope):
    """From a given 4D field of vertical velocities (w=dz/dt), 
    return the 4D field of pressure velocities (omega=dp/dt) with a constant vertical profile 
    in each grid cell, equal to -rho*g*(the mean of w over the lower 200mb); rho~1kg/m3
    
    Args:
        ps : surface pressure; 3D xarray with dimensions latitude, longitude, time
        wupslope : vertical velocity; 4D xarray with dimensions latitude, longitude, level, time

    Returns:
        omega:  pressure velocity; 4D xarray with dimensions latitude, longitude, level, time
    """
    mask = 1* (100.*wupslope.level <= ps)*(100.*wupslope.level >= ps-20000)
    wmasked = wupslope*mask
    omegamean = -1.*9.81*wmasked.sum("level")/mask.sum("level") #omega=-rho*g*w
    omega = omegamean.expand_dims({"level":np.array(wmasked.level)})
    return omega

def upslope_omega_sin(ps,wupslope):
    """From a given 4D field of vertical velocities (w=dz/dt), 
    return the 4D field of pressure velocities (omega=dp/dt) with a sinusoidal vertical profile 
    in each grid cell. The sinusoidal profile is scaled so that its mean over the lower 200mb is 
    equal to the mean pressure velocity corresponding to the input (see upslope_omega_mean200) 
    over the lower 200mb.
    
    Args:
        ps : surface pressure; 3D xarray with dimensions latitude, longitude, time
        wupslope : vertical velocity; 4D xarray with dimensions latitude, longitude, level, time

    Returns:
        omega:  pressure velocity; 4D xarray with dimensions latitude, longitude, level, time
    """
    omegamean200 = upslope_omega_mean200(ps,wupslope)
    mask2 = 1* (100.*omegamean200.level <= ps)*(omegamean200.level >= 200)
    omega =omegamean200* (ps-20000)/(10000*np.pi) * np.sin(np.pi*100*(omegamean200.level-200)/(ps-20000))*mask2    
    return omega

def cape_w(ds,lonlat):
    """compute the vertical velocity due to convection (w = dz/dt) for days in ds, inside lonlat box, 
    using a basic scaling w ~ sqrt(CAPE/2).
    
    Args:
        ds : list of days for which the calculation should be processed; format "YYYYMMDD" (string)
        lonlat : list, [lon1, lon2, lat1, lat2] specifying the box on which to perform calculation. 
            NOTE THAT 0 <= lon1 < lon2 <= 360 and -90 <= lat1 < lat2 <= 90

    Returns:
        w : CAPE vertical velocity; 3D xarray with dimensions latitude, longitude, time
    """

    cape = retrieve_era5_sfc(ds,lonlat,"128_059_cape")
    w = np.sqrt(cape/2.)
    return w

def gz_from_p(p):
    """From a given pressure (in Pa), return the associated geopotential height gz 
    (approximate, assuming dry adiabatic lapse rate and T_surf = 287 K)"""
    gz = 1e3*287*(1-(p/100000)**0.287)
    return gz

def blqe_w(ps,ub,vb,temp_2m,temp,q,gz,lonlat):
    """Compute the vertical velocity due to convection (w = dz/dt) inside lonlat box, 
    using the boundary layer quasi-equilibrium hypothesis. This supposes u.grad(h_b) = w_d(h_mid-h_b)/b,
    where h_b is the boundary layer MSE (taken w/ 2m temperature, z=250m and q(ps) ), h_mid is the 
    mid-troposphere MSE (taken as 500hPa MSE), w_d is the downward velocity outside of 
    clouds, and b is the subcloud layer height (taken as b=500m)
        Denoting by sigma the fractional area of convective clouds, and w_u the upward velocity inside
    clouds, sigma*w_u = (1-sigma)*w_d ~ w_d if sigma << 1; so the mean convective upward velocity is 
    taken as w = w_d
    
    Args:
        ps : surface pressure; 3D xarray with dimensions latitude, longitude, time
        ub : zonal velocity in the boundary layer, 3D xarray with dimensions latitude, longitude, time
        vb : meridional velocity in the boundary layer, 3D xarray with dimensions latitude, longitude, time
        temp_2m : 2m temperature; 3D xarray with dimensions latitude, longitude, time
        temp : temperature, 4D xarray with dimensions latitude, longitude, level, time
        q : specific humidity, 4D xarray with dimensions latitude, longitude, level, time
        gz : geopotential height, 4D xarray with dimensions latitude, longitude, level, time
        lonlat : list, [lon1, lon2, lat1, lat2] specifying the box on which to perform calculation. 
            NOTE THAT 0 <= lon1 < lon2 <= 360 and -90 <= lat1 < lat2 <= 90
            
    Returns:
        w : convective vertical velocity; 3D xarray with dimensions latitude, longitude, time
    """
    Lv = 2257e3 #latent heat of vaporization of water, 100°C, in J/kg
    cp = 1e3 #specific heat at constant pressure of air, 100°C, in J/K/kg
    b = 500 # subcloud layer height, in m

    # Compute Boundary layer MSE (h_b)
    q_surf = q.sel(level=ps/100,method="nearest",drop=True)
    gz_surf = gz.sel(level=ps/100 - 25,method="nearest",drop=True)
    h_b = gz_surf+cp*temp_2m+Lv*q_surf
    
    # Compute upper-tropospheric (200-400hPa mean) saturation MSE (hsat_up)
    temp_up = temp.sel(level=slice(200.,400.))
    gz_up = gz.sel(level=slice(200.,400.))
    qsat_up = qsat(temp_up,temp_up.level)
    hsat_up_vdist = gz_up+cp*temp_up+Lv*qsat_up
    hsat_up=hsat_up_vdist.mean("level")
    
    # Compute 500hPa MSE (h_mid)
    temp_500 = temp.sel(level=500,method="nearest",drop=True)
    q_500 = q.sel(level=500,method="nearest",drop=True)
    gz_500 = gz.sel(level=500,method="nearest",drop=True)
    h_mid = gz_500 + cp*temp_500+Lv*q_500
    
    ddxh_b = h_b.differentiate("longitude")/(m_per_degreelat*coslat)
    ddyh_b = h_b.differentiate("latitude")/m_per_degreelat   
    
    ddxorog1=ddxorog_s.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).isel(time=0)
    ddyorog1=ddyorog_s.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2])).isel(time=0) 
    
    ugradhb = ub*ddxh_b*np.cos(ddxorog1)**2+vb*ddyh_b*np.cos(ddyorog1)**2
    
    wd = np.maximum(0.,b*ugradhb/(h_mid-h_b))*(hsat_up<h_b)
    w=wd
    return w


# ==================================================================
# OMEGA700*qs(surface) PRECIPITATION MODEL, APPLIED TO ERA5 DATA
# ==================================================================

def humidsat(t,p):
    """computes saturation vapor pressure (esat), saturation specific humidity (qsat),
    and saturation mixing ratio (rsat) given inputs temperature (t) in K and
    pressure (p) in hPa.
    
    these are all computed using the modified Tetens-like formulae given by
    Buck (1981, J. Appl. Meteorol.)
    for vapor pressure over liquid water at temperatures over 0 C, and for
    vapor pressure over ice at temperatures below -23 C, and a quadratic
    polynomial interpolation for intermediate temperatures."""
    
    tc=t-273.16
    tice=-23
    t0=0
    Rd=287.04
    Rv=461.5
    epsilon=Rd/Rv


    # first compute saturation vapor pressure over water
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    #alternatively don't use enhancement factor for non-ideal gas correction
    #ewat=6.1121*exp(17.502*tc/(240.97+tc));
    #eice=6.1115*exp(22.452*tc/(272.55+tc));
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))*((tc-tice)/(t0-tice))

    esat=(tc<tice)*eice + (tc>t0)*ewat + (tc>tice)*(tc<t0)*eint

    #now convert vapor pressure to specific humidity and mixing ratio
    rsat=epsilon*esat/(p-esat);
    qsat=epsilon*esat/(p-esat*(1-epsilon));
    
    return esat,qsat,rsat

def qsat(t,p):
    _,q,_=humidsat(t,p)
    return q

def qsat_surface(ds,lonlat):
    if type(ds)!=list:
        raise TypeError("first argument must be a list of days")
    qss=[]
    for d in ds:
        monthbegin = d[:-2]+"01"
        t = xr.open_dataset(glob.glob(CMIP6_FOLDER+"ERA5/e5.oper.an.sfc/*/e5.oper.an.sfc.128_167_2t.*.%s00_*.nc"%monthbegin)[0]).VAR_2T
        p = xr.open_dataset(glob.glob(CMIP6_FOLDER+"ERA5/e5.oper.an.sfc/*/e5.oper.an.sfc.128_134_sp.*.%s00_*.nc"%monthbegin)[0]).SP
        t1 = t.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]),time=d).isel(time=range(0,24,timedisc))
        p1 = p.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]),time=d).isel(time=range(0,24,timedisc))

        qs = qsat(t1,p1/100)
        qss.append(qs)
    return xr.concat(qss,'time')

def precip_model_surface(ds,lonlat):
    """compute total precip in a box given by lonlat, for days in ds, according to upslope flow model"""
    
    if type(ds)!=list:
        raise TypeError("first argument must be a list of days")
    
    #omega
    t = time.time()
    omega1 = retrieve_era5_pl(ds,lonlat,'128_135_w',firstlev=25,levdisc=100)#25 is the index of 700hpa
    w2 = upslope_w(ds,lonlat,firstlev=25,levdisc=100)
    print(time.time()-t)
    qs = qsat_surface(ds,lonlat)
    
    pr1 = np.maximum(-omega1*qs,0.)/9.81
    pr2 = np.maximum(0.9*w2*qs,0.)#rho_700hpa ~ 0.9kg/m^3

    omega2m = -0.9*9.81*w2.mean("time")
    
    
    print(time.time()-t)
    
    return pr1.mean("time")*len(ds)*24*3600,pr2.mean("time")*len(ds)*24*3600,omega2m


# ==================================================================
# O'GORMAN & SCHNEIDER PRECIPITATION MODEL, APPLIED TO ERA5 DATA
# ==================================================================

def scaling2(omega, temp, ps):
    """Same as "scaling", but the arguments are sorted by increasing pressure and plevs are already input.
    scaling2 is to be voctorized by xarray"""
    #Pressure levels from era5
    allplevs = 100.*np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70.,
        100., 125., 150., 175., 200., 225., 250., 300., 350., 400.,
        450., 500., 550., 600., 650., 700., 750., 775., 800., 825.,
        850., 875., 900., 925., 950., 975., 1000.])[range(firstlevel,37,leveldisc)]
    
    return scaling(omega[::-1], temp[::-1], allplevs[::-1], ps)

def precip_model_OGorman(ds,lonlat,omega_models):
    """compute total precip in a box given by lonlat, for days in ds
        with O'Gorman & Schneider model
        
        omega_models is a list if int that gives which models to use for omega :
        1 = True ERA5 omega
        2 = naive upslope omega
        3 = upslope omega averaged 200mb over the surface
        4 = upslope omega with sinusoidal model
        5 = upslope omega with sinusoidal model + sqrt(CAPE/2)
        6 = upslope omega with sinusoidal model + BLQE convective scheme, convective updrafts averaged over whole grid cell
        7 = upslope omega with sinusoidal model + BLQE convective scheme, convective updrafts concentrated on 1% of grid cell
        8 = upslope omega with sinusoidal model + simplified Betts-Miller (SBM) convective scheme
        
        Returns a list of the same length as omega_models giving the precip scaling as a lat/lon array of total precip over 5 days, in mm
        """
    if type(ds)!=list:
        raise TypeError("first argument must be a list of days")    
    
    #temp,ps
    t = time.time()
    temp = retrieve_era5_pl(ds,lonlat,'128_130_t')
    ps = retrieve_era5_sfc(ds,lonlat,'128_134_sp')
    print("elapsed time / temperature, sfc pressure : ", time.time()-t) 
    if any(m in omega_models for m in range(2,10)):
        t = time.time()
        wupslope = upslope_w(ds,lonlat)
        print("elapsed time / upslope omega : ", time.time()-t) 
    
    prs=[]
    omegas=[]
    stored_omega=0 # flag value to indicate no blqe omega has been previously computed
    
    for m in omega_models :
        if m==1 : #true omega
            t = time.time()
            omega = retrieve_era5_pl(ds,lonlat,'128_135_w')
            print("elapsed time / ERA5 omega : ", time.time()-t) 
            pr_hourly = xr.apply_ufunc(scaling2,omega,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
        elif m==2 : #upslope omega
            omega = -9.81*wupslope*wupslope.level*100./temp/287.058 #omega=-rho*g*w=-p/RT*g*w
            pr_hourly = xr.apply_ufunc(scaling2,omega,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
        elif m==3 : #upslope omega, averaged 200mb over surface
            omega = upslope_omega_mean200(ps,wupslope)
            pr_hourly = xr.apply_ufunc(scaling2,omega,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
        elif m==4 : #upslope omega, averaged 200mb over surface with sinusoidal profile
            omega = upslope_omega_sin(ps,wupslope)
            pr_hourly = xr.apply_ufunc(scaling2,omega,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
        elif m==5 : #upslope omega, averaged 200mb over surface + sqrt(CAPE/2) with sinusoidal profile 
            wconvective = cape_w(ds,lonlat)
            omega = upslope_omega_sin(ps,wupslope+wconvective)
            pr_hourly = xr.apply_ufunc(scaling2,omega,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
        elif m==6 or m==7 : #upslope omega, averaged 200mb over surface w/ sinusoidal profile + BLQE omega w/ sin profile
            if stored_omega:
                pass
            else :
                t = time.time()
                ub = retrieve_era5_sfc(ds,lonlat,'228_246_100u') # 100m u
                vb = retrieve_era5_sfc(ds,lonlat,'228_247_100v') # 100m v
                temp_2m = retrieve_era5_sfc(ds,lonlat,'128_167_2t')            
                print("elapsed time / ub,vb and 2m temperature : ", time.time()-t)
                
                t = time.time()
                q = retrieve_era5_pl(ds,lonlat,'128_133_q',firstlev=21,levdisc=1)
                print("elapsed time / specific humidity : ", time.time()-t) 
                
                t = time.time()
                gz = retrieve_era5_pl(ds,lonlat,'128_129_z',firstlev=14,levdisc=1)
                print("elapsed time / geopotential height : ", time.time()-t) 
                
                wconvective = blqe_w(ps,ub,vb,temp_2m,temp,q,gz,lonlat)
                
                omega_largescale = upslope_omega_sin(ps,wupslope)
                omega_convective = upslope_omega_sin(ps,wconvective.expand_dims({"level":np.array(wupslope.level)}))
                
                omega = upslope_omega_sin(ps,wupslope+wconvective) 
                #used only if m==6, in which case Wconvective is taken averaged over the whole grid cell
                stored_omega=1
            if m==6:
                pr_hourly = xr.apply_ufunc(scaling2,omega,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
            elif m==7:#Wconvective is concentrated in updrafts representing 1% of the region
                pr_largescale = xr.apply_ufunc(scaling2,omega_largescale,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
                pr_convective = xr.apply_ufunc(scaling2,100.*omega_convective,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
                pr_hourly = pr_largescale+pr_convective/100.
                
        elif m==8: #upslope omega + convective betts-miller omega
            t = time.time()
            temp_2m = retrieve_era5_sfc(ds,lonlat,'128_167_2t')
            q = retrieve_era5_pl(ds,lonlat,'128_133_q',firstlev=21,levdisc=1)
            print("elapsed time / specific humidity and 2m temperature: ", time.time()-t)
            
            omega_largescale = upslope_omega_sin(ps,wupslope)
            pr_largescale = xr.apply_ufunc(scaling2,omega_largescale,temp,ps,input_core_dims=[['level'], ['level'],[]],vectorize=True)
            pr_convective = sbm(ps,temp_2m,q)
            pr_hourly = pr_largescale+pr_convective

        else : 
            print("Unknown model number, please see docstring")
            continue
        
        prs.append(np.maximum(pr_hourly.mean("time")*len(ds)*24*3600,0.))
        #omegas.append(omega.mean("time"))
    
    return prs#,omegas

def retrieve_modeled_precip(subfolder,ds,lonlat,ct,omega_modelnames=[],trmm=1,gpcp=0,era5=0):
    """get the precip data in the $SCRATCH/precipmodel/"subfolder"/pr/ folder as well as TRMM, GPCP and ERA5 precip data
    ds = list of days eg ["20000626","20010730"]
    lonlat = lon/lat bounds eg [80, 90, 25, 30]
    ct = country name eg "nepal"
    
    omega_modelnames is a list of strings that each specify the file name patterns to look for, e.g.
    ["omega700ERA5Xqs","omega700upslopeXqs","omegaERA5Xqinteg.plev19_2"]
    """
    
    n=len(omega_modelnames)
    prs=np.zeros((n+trmm+gpcp+era5,len(ds)))
    
    for j in range(len(ds)) :
        d=ds[j]
        for i in range(n):
            m = omega_modelnames[i]
            prt = xr.open_dataset("/global/cscratch1/sd/qnicolas/precipmodel/%s/pr/pr.%s.%s.%s.nc"%(subfolder,m,d,ct))
            prs[i,j]=spacemean_era5(prt,lonlat).__xarray_dataarray_variable__
        if subfolder == "p99":
            if trmm :
                try :
                    prs[-1-gpcp-era5,j]=spacemean_trmm(precipTRMM1d(d,lonlat),lonlat) 
                except KeyError:
                    print("Warning : TRMM data not found for %s"%d)
                    prs[-1-gpcp-era5,j]=0
            if gpcp :
                prs[-1-era5,j]=spacemean_trmm(precipGPCP1d(d,lonlat),lonlat) 
            if era5 :
                prs[-1,j]=spacemean_era5(precipERA51d(d,lonlat),lonlat)
        elif subfolder == "rx5":
            if trmm :
                try :
                    prs[-1-gpcp-era5,j]=spacemean_trmm(precipTRMM1d(d,lonlat),lonlat) 
                except KeyError:
                    print("Warning : TRMM data not found for %s"%d)
                    prs[-1-gpcp-era5,j]=0
            if gpcp :
                prs[-1-era5,j]=spacemean_trmm(precipGPCP5d(d,lonlat),lonlat) 
            if era5 :
                prs[-1,j]=spacemean_era5(precipERA55d(d,lonlat),lonlat)
        else : 
            raise ValueError("subfolder must be either p99 or rx5")
            
    return prs

def retrieve_omega_rx5(ds,lonlat,ct,omega_modelnames=[]):
    """get the omega data in the $SCRATCH/precipmodel/rx5/omega folder
    omega_modelnames is a list of strings that specify the file name patterns to look for, e.g.
    ["oper.an.pl.128_135_w.ll025sc","upslopeomegamean200","upslopeomegasin"]
    """
    
    if type(ds)!=list:
        raise TypeError("first argument must be a list of days")
    
    n=len(omega_modelnames)
    prs= [[] for i in range(n)]
    
    for d in ds :
        for i in range(n):
            m = omega_modelnames[i]
            omega = xr.open_dataset("/global/cscratch1/sd/qnicolas/precipmodel/rx5/omega/e5.%s.%s.%s.nc"%(m,d,ct))
            prs[i].append(omega)
    return prs