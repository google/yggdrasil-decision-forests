ůÎ
ĺ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
Ą
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ł
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
f
SimpleMLCreateModelResource
model_handle"
	containerstring "
shared_namestring 
á
SimpleMLInferenceOpWithHandle
numerical_features
boolean_features
categorical_int_features'
#categorical_set_int_features_values1
-categorical_set_int_features_row_splits_dim_1	1
-categorical_set_int_features_row_splits_dim_2	
model_handle
dense_predictions
dense_col_representation"
dense_output_dimint(0
Ł
#SimpleMLLoadModelFromPathWithHandle
model_handle
path" 
output_typeslist(string)
 "
file_prefixstring " 
allow_slow_inferencebool(
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
°
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized
"serve*2.16.02unknown8Ź
r
ConstConst*
_output_shapes
:	*
dtype0*9
value0B.	"$˙˙˙˙˙˙˙˙                     
ľ
Const_1Const*
_output_shapes
:	*
dtype0*z
valueqBo	B B
2147483645BPrivateBSelf-emp-not-incB	Local-govB	State-govBSelf-emp-incBFederal-govBWithout-pay
j
Const_2Const*
_output_shapes
:*
dtype0*/
value&B$B B
2147483645BMaleBFemale
`
Const_3Const*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙˙˙˙˙      
p
Const_4Const*
_output_shapes
:*
dtype0*5
value,B*" ˙˙˙˙˙˙˙˙                  
 
Const_5Const*
_output_shapes
:*
dtype0*e
value\BZB B
2147483645BHusbandBNot-in-familyB	Own-childB	UnmarriedBWifeBOther-relative
l
Const_6Const*
_output_shapes
:*
dtype0*1
value(B&"˙˙˙˙˙˙˙˙               

Const_7Const*
_output_shapes
:*
dtype0*^
valueUBSB B
2147483645BWhiteBBlackBAsian-Pac-IslanderBAmer-Indian-EskimoBOther

Const_8Const*
_output_shapes
:*
dtype0*Q
valueHBF"<˙˙˙˙˙˙˙˙                        	   
            
Ť
Const_9Const*
_output_shapes
:*
dtype0*ď
valueĺBâB B
2147483645BProf-specialtyBExec-managerialBCraft-repairBAdm-clericalBSalesBOther-serviceBMachine-op-inspctBTransport-movingBHandlers-cleanersBFarming-fishingBTech-supportBProtective-servBPriv-house-serv
ě
Const_10Const*
_output_shapes
:**
dtype0*Ż
valueĽB˘*B B
2147483645BUnited-StatesBMexicoBPhilippinesBGermanyBCanadaBPuerto-RicoBIndiaBEl-SalvadorBCubaBEnglandBJamaicaBDominican-RepublicBSouthBChinaBItalyBColumbiaB	GuatemalaBJapanBVietnamBTaiwanBPolandBIranBHaitiB	NicaraguaBPortugalBGreeceBPeruBFranceBEcuadorBThailandBCambodiaBLaosBIrelandB
YugoslaviaBTrinadad&TobagoBHondurasBHungaryBHongBScotlandBOutlying-US(Guam-USVI-etc)
ý
Const_11Const*
_output_shapes
:**
dtype0*Ŕ
valueśBł*"¨˙˙˙˙˙˙˙˙                        	   
                                                                      !   "   #   $   %   &   '   (   
u
Const_12Const*
_output_shapes
:	*
dtype0*9
value0B.	"$˙˙˙˙˙˙˙˙                     
Ë
Const_13Const*
_output_shapes
:	*
dtype0*
valueB	B B
2147483645BMarried-civ-spouseBNever-marriedBDivorcedBWidowedB	SeparatedBMarried-spouse-absentBMarried-AF-spouse

Const_14Const*
_output_shapes
:*
dtype0*]
valueTBR"H˙˙˙˙˙˙˙˙                        	   
                     
÷
Const_15Const*
_output_shapes
:*
dtype0*ş
value°B­B B
2147483645BHS-gradBSome-collegeB	BachelorsBMastersB	Assoc-vocB11thB
Assoc-acdmB10thB7th-8thBProf-schoolB9thB12thB	DoctorateB5th-6thB1st-4thB	Preschool
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 

VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
z
Variable/AssignAssignVariableOpVariableasset_path_initializer*&
 _has_manual_control_dependencies(*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
J
Const_16Const*
_output_shapes
: *
dtype0*
value	B : 
J
Const_17Const*
_output_shapes
: *
dtype0*
value	B : 
J
Const_18Const*
_output_shapes
: *
dtype0*
value	B : 
J
Const_19Const*
_output_shapes
: *
dtype0*
value	B : 
J
Const_20Const*
_output_shapes
: *
dtype0*
value	B : 
J
Const_21Const*
_output_shapes
: *
dtype0*
value	B : 
J
Const_22Const*
_output_shapes
: *
dtype0*
value	B : 
J
Const_23Const*
_output_shapes
: *
dtype0*
value	B : 
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 

Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer_1*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
Y
asset_path_initializer_2Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape: *
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 

Variable_2/AssignAssignVariableOp
Variable_2asset_path_initializer_2*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
Y
asset_path_initializer_3Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape: *
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 

Variable_3/AssignAssignVariableOp
Variable_3asset_path_initializer_3*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
Y
asset_path_initializer_4Placeholder*
_output_shapes
: *
dtype0*
shape: 
¤

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape: *
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 

Variable_4/AssignAssignVariableOp
Variable_4asset_path_initializer_4*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1671*
value_dtype0
n
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1665*
value_dtype0
n
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1659*
value_dtype0
n
hash_table_3HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1653*
value_dtype0
n
hash_table_4HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1647*
value_dtype0
n
hash_table_5HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1641*
value_dtype0
n
hash_table_6HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1635*
value_dtype0
n
hash_table_7HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1629*
value_dtype0

SimpleMLCreateModelResourceSimpleMLCreateModelResource*
_output_shapes
: *E
shared_name64simple_ml_model_3d1fefc4-02b3-4e10-81a2-b58d8bd459cd

learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	


is_trainedVarHandleOp*
_output_shapes
: *

debug_nameis_trained/*
dtype0
*
shape: *
shared_name
is_trained
a
is_trained/Read/ReadVariableOpReadVariableOp
is_trained*
_output_shapes
: *
dtype0

n
serving_default_agePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_capital_gainPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_capital_lossPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
t
serving_default_educationPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_education_numPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
q
serving_default_fnlwgtPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_hours_per_weekPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_marital_statusPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_native_countryPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
u
serving_default_occupationPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
o
serving_default_racePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_relationshipPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
n
serving_default_sexPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
t
serving_default_workclassPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

StatefulPartitionedCallStatefulPartitionedCallserving_default_ageserving_default_capital_gainserving_default_capital_lossserving_default_educationserving_default_education_numserving_default_fnlwgtserving_default_hours_per_weekserving_default_marital_statusserving_default_native_countryserving_default_occupationserving_default_raceserving_default_relationshipserving_default_sexserving_default_workclass
hash_tableConst_17hash_table_7Const_16hash_table_6Const_23hash_table_4Const_22hash_table_2Const_21hash_table_3Const_20hash_table_1Const_19hash_table_5Const_18SimpleMLCreateModelResource**
Tin#
!2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_2446
a
ReadVariableOpReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
×
StatefulPartitionedCall_1StatefulPartitionedCallReadVariableOpSimpleMLCreateModelResource*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2457
Í
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_7Const_15Const_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2472
Í
StatefulPartitionedCall_3StatefulPartitionedCallhash_table_6Const_13Const_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2487
Í
StatefulPartitionedCall_4StatefulPartitionedCallhash_table_5Const_10Const_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2502
Ë
StatefulPartitionedCall_5StatefulPartitionedCallhash_table_4Const_9Const_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2517
Ë
StatefulPartitionedCall_6StatefulPartitionedCallhash_table_3Const_7Const_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2532
Ë
StatefulPartitionedCall_7StatefulPartitionedCallhash_table_2Const_5Const_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2547
Ë
StatefulPartitionedCall_8StatefulPartitionedCallhash_table_1Const_2Const_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2562
Ç
StatefulPartitionedCall_9StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__initializer_2577
ę
NoOpNoOp^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6^StatefulPartitionedCall_7^StatefulPartitionedCall_8^StatefulPartitionedCall_9^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign
Ę"
Const_24Const"/device:CPU:0*
_output_shapes
: *
dtype0*"
valueř!Bő! Bî!
Ą
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

_multitask
	_is_trained

_learner_params
	_features
	optimizer
loss
_models
_build_normalized_inputs
_finalize_predictions
call
call_get_leaves
yggdrasil_model_path_tensor

signatures*

	0*
* 
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
}
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15* 
* 
JD
VARIABLE_VALUE
is_trained&_is_trained/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
O
&
_variables
'_iterations
(_learning_rate
)_update_step_xla*
* 
	
*0* 

+trace_0* 

,trace_0* 

-trace_0* 
* 

.trace_0* 

/serving_default* 

	0*
* 
* 
* 
* 
}
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15* 
}
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15* 
}
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15* 
}
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15* 
* 
* 
* 
* 
* 
* 
* 
* 

'0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
+
0_input_builder
1_compiled_model* 
* 
* 
}
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15* 

2	capture_0* 
}
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15* 
P
3_feature_name_to_idx
4	_init_ops
#5categorical_str_to_int_hashmaps* 
S
6_model_loader
7_create_resource
8_initialize
9_destroy_resource* 
* 
* 
* 
}
:	education
;marital_status
<native_country
=
occupation
>race
?relationship
@sex
A	workclass* 
5
B_output_types
C
_all_files
2
_done_file* 

Dtrace_0* 

Etrace_0* 

Ftrace_0* 
R
G_initializer
H_create_resource
I_initialize
J_destroy_resource* 
R
K_initializer
L_create_resource
M_initialize
N_destroy_resource* 
R
O_initializer
P_create_resource
Q_initialize
R_destroy_resource* 
R
S_initializer
T_create_resource
U_initialize
V_destroy_resource* 
R
W_initializer
X_create_resource
Y_initialize
Z_destroy_resource* 
R
[_initializer
\_create_resource
]_initialize
^_destroy_resource* 
R
__initializer
`_create_resource
a_initialize
b_destroy_resource* 
R
c_initializer
d_create_resource
e_initialize
f_destroy_resource* 
* 
%
g0
21
h2
i3
j4* 
* 

2	capture_0* 
* 
* 

ktrace_0* 

ltrace_0* 

mtrace_0* 
* 

ntrace_0* 

otrace_0* 

ptrace_0* 
* 

qtrace_0* 

rtrace_0* 

strace_0* 
* 

ttrace_0* 

utrace_0* 

vtrace_0* 
* 

wtrace_0* 

xtrace_0* 

ytrace_0* 
* 

ztrace_0* 

{trace_0* 

|trace_0* 
* 

}trace_0* 

~trace_0* 

trace_0* 
* 

trace_0* 

trace_0* 

trace_0* 
* 
* 
* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
"
	capture_1
	capture_2* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ć
StatefulPartitionedCall_10StatefulPartitionedCallsaver_filename
is_trained	iterationlearning_rateConst_24*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_2697
ž
StatefulPartitionedCall_11StatefulPartitionedCallsaver_filename
is_trained	iterationlearning_rate*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_2715Ń

ö
__inference__initializer_25027
3key_value_init1640_lookuptableimportv2_table_handle/
+key_value_init1640_lookuptableimportv2_keys1
-key_value_init1640_lookuptableimportv2_values
identity˘&key_value_init1640/LookupTableImportV2ű
&key_value_init1640/LookupTableImportV2LookupTableImportV23key_value_init1640_lookuptableimportv2_table_handle+key_value_init1640_lookuptableimportv2_keys-key_value_init1640_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1640/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :*:*2P
&key_value_init1640/LookupTableImportV2&key_value_init1640/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:*: 

_output_shapes
:*
Ź
­
"__inference_signature_wrapper_2446
age	
capital_gain	
capital_loss	
	education
education_num	

fnlwgt	
hours_per_week	
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15**
Tin#
!2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_2042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameage:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_gain:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_loss:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	education:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameeducation_num:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namefnlwgt:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namehours_per_week:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemarital_status:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namenative_country:O	K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
occupation:I
E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namerace:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerelationship:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namesex:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	workclass:$ 

_user_specified_name2410:

_output_shapes
: :$ 

_user_specified_name2414:

_output_shapes
: :$ 

_user_specified_name2418:

_output_shapes
: :$ 

_user_specified_name2422:

_output_shapes
: :$ 

_user_specified_name2426:

_output_shapes
: :$ 

_user_specified_name2430:

_output_shapes
: :$ 

_user_specified_name2434:

_output_shapes
: :$ 

_user_specified_name2438:

_output_shapes
: :$ 

_user_specified_name2442

+
__inference__destroyer_2551
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
é
ţ
 __inference__traced_restore_2715
file_prefix%
assignvariableop_is_trained:
 &
assignvariableop_1_iteration:	 *
 assignvariableop_2_learning_rate: 

identity_4˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Â
value¸BľB&_is_trained/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHx
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B ˛
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
2
	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0
*
_output_shapes
:Ž
AssignVariableOpAssignVariableOpassignvariableop_is_trainedIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0
]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:ł
AssignVariableOp_1AssignVariableOpassignvariableop_1_iterationIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_2AssignVariableOp assignvariableop_2_learning_rateIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: a
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_user_specified_name
is_trained:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate
Ź
J
__inference__creator_2450
identity˘SimpleMLCreateModelResource
SimpleMLCreateModelResourceSimpleMLCreateModelResource*
_output_shapes
: *E
shared_name64simple_ml_model_3d1fefc4-02b3-4e10-81a2-b58d8bd459cdh
IdentityIdentity*SimpleMLCreateModelResource:model_handle:0^NoOp*
T0*
_output_shapes
: @
NoOpNoOp^SimpleMLCreateModelResource*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2:
SimpleMLCreateModelResourceSimpleMLCreateModelResource
Ć#
˙
__inference__wrapped_model_2042
age	
capital_gain	
capital_loss	
	education
education_num	

fnlwgt	
hours_per_week	
marital_status
native_country

occupation
race
relationship
sex
	workclass'
#gradient_boosted_trees_model_1_2006'
#gradient_boosted_trees_model_1_2008'
#gradient_boosted_trees_model_1_2010'
#gradient_boosted_trees_model_1_2012'
#gradient_boosted_trees_model_1_2014'
#gradient_boosted_trees_model_1_2016'
#gradient_boosted_trees_model_1_2018'
#gradient_boosted_trees_model_1_2020'
#gradient_boosted_trees_model_1_2022'
#gradient_boosted_trees_model_1_2024'
#gradient_boosted_trees_model_1_2026'
#gradient_boosted_trees_model_1_2028'
#gradient_boosted_trees_model_1_2030'
#gradient_boosted_trees_model_1_2032'
#gradient_boosted_trees_model_1_2034'
#gradient_boosted_trees_model_1_2036'
#gradient_boosted_trees_model_1_2038
identity˘6gradient_boosted_trees_model_1/StatefulPartitionedCallÝ
6gradient_boosted_trees_model_1/StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass#gradient_boosted_trees_model_1_2006#gradient_boosted_trees_model_1_2008#gradient_boosted_trees_model_1_2010#gradient_boosted_trees_model_1_2012#gradient_boosted_trees_model_1_2014#gradient_boosted_trees_model_1_2016#gradient_boosted_trees_model_1_2018#gradient_boosted_trees_model_1_2020#gradient_boosted_trees_model_1_2022#gradient_boosted_trees_model_1_2024#gradient_boosted_trees_model_1_2026#gradient_boosted_trees_model_1_2028#gradient_boosted_trees_model_1_2030#gradient_boosted_trees_model_1_2032#gradient_boosted_trees_model_1_2034#gradient_boosted_trees_model_1_2036#gradient_boosted_trees_model_1_2038**
Tin#
!2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *
fR
__inference_call_2005
IdentityIdentity?gradient_boosted_trees_model_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[
NoOpNoOp7^gradient_boosted_trees_model_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 2p
6gradient_boosted_trees_model_1/StatefulPartitionedCall6gradient_boosted_trees_model_1/StatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameage:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_gain:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_loss:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	education:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameeducation_num:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namefnlwgt:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namehours_per_week:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemarital_status:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namenative_country:O	K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
occupation:I
E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namerace:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerelationship:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namesex:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	workclass:$ 

_user_specified_name2006:

_output_shapes
: :$ 

_user_specified_name2010:

_output_shapes
: :$ 

_user_specified_name2014:

_output_shapes
: :$ 

_user_specified_name2018:

_output_shapes
: :$ 

_user_specified_name2022:

_output_shapes
: :$ 

_user_specified_name2026:

_output_shapes
: :$ 

_user_specified_name2030:

_output_shapes
: :$ 

_user_specified_name2034:

_output_shapes
: :$ 

_user_specified_name2038
Š
9
__inference__creator_2480
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1635*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table

+
__inference__destroyer_2491
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 

ö
__inference__initializer_25327
3key_value_init1652_lookuptableimportv2_table_handle/
+key_value_init1652_lookuptableimportv2_keys1
-key_value_init1652_lookuptableimportv2_values
identity˘&key_value_init1652/LookupTableImportV2ű
&key_value_init1652/LookupTableImportV2LookupTableImportV23key_value_init1652_lookuptableimportv2_table_handle+key_value_init1652_lookuptableimportv2_keys-key_value_init1652_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1652/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2P
&key_value_init1652/LookupTableImportV2&key_value_init1652/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:

+
__inference__destroyer_2536
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 

ö
__inference__initializer_25777
3key_value_init1670_lookuptableimportv2_table_handle/
+key_value_init1670_lookuptableimportv2_keys1
-key_value_init1670_lookuptableimportv2_values
identity˘&key_value_init1670/LookupTableImportV2ű
&key_value_init1670/LookupTableImportV2LookupTableImportV23key_value_init1670_lookuptableimportv2_table_handle+key_value_init1670_lookuptableimportv2_keys-key_value_init1670_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1670/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :	:	2P
&key_value_init1670/LookupTableImportV2&key_value_init1670/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:	: 

_output_shapes
:	

ö
__inference__initializer_24727
3key_value_init1628_lookuptableimportv2_table_handle/
+key_value_init1628_lookuptableimportv2_keys1
-key_value_init1628_lookuptableimportv2_values
identity˘&key_value_init1628/LookupTableImportV2ű
&key_value_init1628/LookupTableImportV2LookupTableImportV23key_value_init1628_lookuptableimportv2_table_handle+key_value_init1628_lookuptableimportv2_keys-key_value_init1628_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1628/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2P
&key_value_init1628/LookupTableImportV2&key_value_init1628/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:

ö
__inference__initializer_25177
3key_value_init1646_lookuptableimportv2_table_handle/
+key_value_init1646_lookuptableimportv2_keys1
-key_value_init1646_lookuptableimportv2_values
identity˘&key_value_init1646/LookupTableImportV2ű
&key_value_init1646/LookupTableImportV2LookupTableImportV23key_value_init1646_lookuptableimportv2_table_handle+key_value_init1646_lookuptableimportv2_keys-key_value_init1646_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1646/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2P
&key_value_init1646/LookupTableImportV2&key_value_init1646/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:

+
__inference__destroyer_2506
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Š
9
__inference__creator_2570
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1671*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table

+
__inference__destroyer_2566
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
˝
Z
,__inference_yggdrasil_model_path_tensor_2393
staticregexreplace_input
identity
StaticRegexReplaceStaticRegexReplacestaticregexreplace_input*
_output_shapes
: *!
pattern3bb60e6d9306498ddone*
rewrite R
IdentityIdentityStaticRegexReplace:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 

+
__inference__destroyer_2461
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
˛
ž
__inference__initializer_2457
staticregexreplace_input>
:simple_ml_simplemlloadmodelfrompathwithhandle_model_handle
identity˘-simple_ml/SimpleMLLoadModelFromPathWithHandle
StaticRegexReplaceStaticRegexReplacestaticregexreplace_input*
_output_shapes
: *!
pattern3bb60e6d9306498ddone*
rewrite ć
-simple_ml/SimpleMLLoadModelFromPathWithHandle#SimpleMLLoadModelFromPathWithHandle:simple_ml_simplemlloadmodelfrompathwithhandle_model_handleStaticRegexReplace:output:0*
_output_shapes
 *!
file_prefix3bb60e6d9306498dG
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^simple_ml/SimpleMLLoadModelFromPathWithHandle*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2^
-simple_ml/SimpleMLLoadModelFromPathWithHandle-simple_ml/SimpleMLLoadModelFromPathWithHandle: 

_output_shapes
: :,(
&
_user_specified_namemodel_handle

ö
__inference__initializer_25627
3key_value_init1664_lookuptableimportv2_table_handle/
+key_value_init1664_lookuptableimportv2_keys1
-key_value_init1664_lookuptableimportv2_values
identity˘&key_value_init1664/LookupTableImportV2ű
&key_value_init1664/LookupTableImportV2LookupTableImportV23key_value_init1664_lookuptableimportv2_table_handle+key_value_init1664_lookuptableimportv2_keys-key_value_init1664_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1664/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2P
&key_value_init1664/LookupTableImportV2&key_value_init1664/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
şD
Ś
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2107
age	
capital_gain	
capital_loss	
	education
education_num	

fnlwgt	
hours_per_week	
marital_status
native_country

occupation
race
relationship
sex
	workclass.
*none_lookup_lookuptablefindv2_table_handle/
+none_lookup_lookuptablefindv2_default_value0
,none_lookup_1_lookuptablefindv2_table_handle1
-none_lookup_1_lookuptablefindv2_default_value0
,none_lookup_2_lookuptablefindv2_table_handle1
-none_lookup_2_lookuptablefindv2_default_value0
,none_lookup_3_lookuptablefindv2_table_handle1
-none_lookup_3_lookuptablefindv2_default_value0
,none_lookup_4_lookuptablefindv2_table_handle1
-none_lookup_4_lookuptablefindv2_default_value0
,none_lookup_5_lookuptablefindv2_table_handle1
-none_lookup_5_lookuptablefindv2_default_value0
,none_lookup_6_lookuptablefindv2_table_handle1
-none_lookup_6_lookuptablefindv2_default_value0
,none_lookup_7_lookuptablefindv2_table_handle1
-none_lookup_7_lookuptablefindv2_default_value
inference_op_model_handle
identity˘None_Lookup/LookupTableFindV2˘None_Lookup_1/LookupTableFindV2˘None_Lookup_2/LookupTableFindV2˘None_Lookup_3/LookupTableFindV2˘None_Lookup_4/LookupTableFindV2˘None_Lookup_5/LookupTableFindV2˘None_Lookup_6/LookupTableFindV2˘None_Lookup_7/LookupTableFindV2˘inference_op
PartitionedCallPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass*
Tin
2						*
Tout
2*
_collective_manager_ids
 *č
_output_shapesŐ
Ň:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference__build_normalized_inputs_1946â
None_Lookup/LookupTableFindV2LookupTableFindV2*none_lookup_lookuptablefindv2_table_handlePartitionedCall:output:13+none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_1/LookupTableFindV2LookupTableFindV2,none_lookup_1_lookuptablefindv2_table_handlePartitionedCall:output:3-none_lookup_1_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_2/LookupTableFindV2LookupTableFindV2,none_lookup_2_lookuptablefindv2_table_handlePartitionedCall:output:7-none_lookup_2_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_3/LookupTableFindV2LookupTableFindV2,none_lookup_3_lookuptablefindv2_table_handlePartitionedCall:output:9-none_lookup_3_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_4/LookupTableFindV2LookupTableFindV2,none_lookup_4_lookuptablefindv2_table_handlePartitionedCall:output:11-none_lookup_4_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_5/LookupTableFindV2LookupTableFindV2,none_lookup_5_lookuptablefindv2_table_handlePartitionedCall:output:10-none_lookup_5_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_6/LookupTableFindV2LookupTableFindV2,none_lookup_6_lookuptablefindv2_table_handlePartitionedCall:output:12-none_lookup_6_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_7/LookupTableFindV2LookupTableFindV2,none_lookup_7_lookuptablefindv2_table_handlePartitionedCall:output:8-none_lookup_7_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  ¤
stack_1Pack(None_Lookup_1/LookupTableFindV2:values:0(None_Lookup_2/LookupTableFindV2:values:0(None_Lookup_7/LookupTableFindV2:values:0(None_Lookup_3/LookupTableFindV2:values:0(None_Lookup_5/LookupTableFindV2:values:0(None_Lookup_4/LookupTableFindV2:values:0(None_Lookup_6/LookupTableFindV2:values:0&None_Lookup/LookupTableFindV2:values:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisX
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0stack_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dim×
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference__finalize_predictions_2002i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ż
NoOpNoOp^None_Lookup/LookupTableFindV2 ^None_Lookup_1/LookupTableFindV2 ^None_Lookup_2/LookupTableFindV2 ^None_Lookup_3/LookupTableFindV2 ^None_Lookup_4/LookupTableFindV2 ^None_Lookup_5/LookupTableFindV2 ^None_Lookup_6/LookupTableFindV2 ^None_Lookup_7/LookupTableFindV2^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 2>
None_Lookup/LookupTableFindV2None_Lookup/LookupTableFindV22B
None_Lookup_1/LookupTableFindV2None_Lookup_1/LookupTableFindV22B
None_Lookup_2/LookupTableFindV2None_Lookup_2/LookupTableFindV22B
None_Lookup_3/LookupTableFindV2None_Lookup_3/LookupTableFindV22B
None_Lookup_4/LookupTableFindV2None_Lookup_4/LookupTableFindV22B
None_Lookup_5/LookupTableFindV2None_Lookup_5/LookupTableFindV22B
None_Lookup_6/LookupTableFindV2None_Lookup_6/LookupTableFindV22B
None_Lookup_7/LookupTableFindV2None_Lookup_7/LookupTableFindV22
inference_opinference_op:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameage:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_gain:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_loss:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	education:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameeducation_num:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namefnlwgt:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namehours_per_week:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemarital_status:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namenative_country:O	K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
occupation:I
E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namerace:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerelationship:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namesex:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	workclass:,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_namemodel_handle
Ű

&__inference__finalize_predictions_2323!
predictions_dense_predictions(
$predictions_dense_col_representation
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ű
strided_sliceStridedSlicepredictions_dense_predictionsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

begin_mask*
end_mask^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:˙˙˙˙˙˙˙˙˙::f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namepredictions_dense_predictions:`\

_output_shapes
:
>
_user_specified_name&$predictions_dense_col_representation
Š
9
__inference__creator_2525
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1653*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table

Č
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2276
age	
capital_gain	
capital_loss	
	education
education_num	

fnlwgt	
hours_per_week	
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity˘StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15**
Tin#
!2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2172o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameage:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_gain:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_loss:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	education:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameeducation_num:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namefnlwgt:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namehours_per_week:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemarital_status:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namenative_country:O	K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
occupation:I
E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namerace:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerelationship:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namesex:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	workclass:$ 

_user_specified_name2240:

_output_shapes
: :$ 

_user_specified_name2244:

_output_shapes
: :$ 

_user_specified_name2248:

_output_shapes
: :$ 

_user_specified_name2252:

_output_shapes
: :$ 

_user_specified_name2256:

_output_shapes
: :$ 

_user_specified_name2260:

_output_shapes
: :$ 

_user_specified_name2264:

_output_shapes
: :$ 

_user_specified_name2268:

_output_shapes
: :$ 

_user_specified_name2272

Ó
)__inference__build_normalized_inputs_1946

inputs	
	inputs_10	
	inputs_11	
inputs_3
inputs_4	
inputs_2	
	inputs_12	
inputs_5
	inputs_13
inputs_6
inputs_8
inputs_7
inputs_9
inputs_1
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13Q
CastCastinputs*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙U
Cast_1Castinputs_2*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙U
Cast_2Castinputs_4*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_3Cast	inputs_10*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_4Cast	inputs_11*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_5Cast	inputs_12*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙L
IdentityIdentityCast:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_1Identity
Cast_3:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_2Identity
Cast_4:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_3Identityinputs_3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_4Identity
Cast_2:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_5Identity
Cast_1:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_6Identity
Cast_5:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_7Identityinputs_5*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_8Identity	inputs_13*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙N

Identity_9Identityinputs_6*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_10Identityinputs_8*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_11Identityinputs_7*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_12Identityinputs_9*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_13Identityinputs_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*ç
_input_shapesŐ
Ň:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Š
9
__inference__creator_2540
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1659*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
şD
Ś
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2172
age	
capital_gain	
capital_loss	
	education
education_num	

fnlwgt	
hours_per_week	
marital_status
native_country

occupation
race
relationship
sex
	workclass.
*none_lookup_lookuptablefindv2_table_handle/
+none_lookup_lookuptablefindv2_default_value0
,none_lookup_1_lookuptablefindv2_table_handle1
-none_lookup_1_lookuptablefindv2_default_value0
,none_lookup_2_lookuptablefindv2_table_handle1
-none_lookup_2_lookuptablefindv2_default_value0
,none_lookup_3_lookuptablefindv2_table_handle1
-none_lookup_3_lookuptablefindv2_default_value0
,none_lookup_4_lookuptablefindv2_table_handle1
-none_lookup_4_lookuptablefindv2_default_value0
,none_lookup_5_lookuptablefindv2_table_handle1
-none_lookup_5_lookuptablefindv2_default_value0
,none_lookup_6_lookuptablefindv2_table_handle1
-none_lookup_6_lookuptablefindv2_default_value0
,none_lookup_7_lookuptablefindv2_table_handle1
-none_lookup_7_lookuptablefindv2_default_value
inference_op_model_handle
identity˘None_Lookup/LookupTableFindV2˘None_Lookup_1/LookupTableFindV2˘None_Lookup_2/LookupTableFindV2˘None_Lookup_3/LookupTableFindV2˘None_Lookup_4/LookupTableFindV2˘None_Lookup_5/LookupTableFindV2˘None_Lookup_6/LookupTableFindV2˘None_Lookup_7/LookupTableFindV2˘inference_op
PartitionedCallPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass*
Tin
2						*
Tout
2*
_collective_manager_ids
 *č
_output_shapesŐ
Ň:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference__build_normalized_inputs_1946â
None_Lookup/LookupTableFindV2LookupTableFindV2*none_lookup_lookuptablefindv2_table_handlePartitionedCall:output:13+none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_1/LookupTableFindV2LookupTableFindV2,none_lookup_1_lookuptablefindv2_table_handlePartitionedCall:output:3-none_lookup_1_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_2/LookupTableFindV2LookupTableFindV2,none_lookup_2_lookuptablefindv2_table_handlePartitionedCall:output:7-none_lookup_2_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_3/LookupTableFindV2LookupTableFindV2,none_lookup_3_lookuptablefindv2_table_handlePartitionedCall:output:9-none_lookup_3_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_4/LookupTableFindV2LookupTableFindV2,none_lookup_4_lookuptablefindv2_table_handlePartitionedCall:output:11-none_lookup_4_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_5/LookupTableFindV2LookupTableFindV2,none_lookup_5_lookuptablefindv2_table_handlePartitionedCall:output:10-none_lookup_5_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_6/LookupTableFindV2LookupTableFindV2,none_lookup_6_lookuptablefindv2_table_handlePartitionedCall:output:12-none_lookup_6_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_7/LookupTableFindV2LookupTableFindV2,none_lookup_7_lookuptablefindv2_table_handlePartitionedCall:output:8-none_lookup_7_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  ¤
stack_1Pack(None_Lookup_1/LookupTableFindV2:values:0(None_Lookup_2/LookupTableFindV2:values:0(None_Lookup_7/LookupTableFindV2:values:0(None_Lookup_3/LookupTableFindV2:values:0(None_Lookup_5/LookupTableFindV2:values:0(None_Lookup_4/LookupTableFindV2:values:0(None_Lookup_6/LookupTableFindV2:values:0&None_Lookup/LookupTableFindV2:values:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisX
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0stack_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dim×
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference__finalize_predictions_2002i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ż
NoOpNoOp^None_Lookup/LookupTableFindV2 ^None_Lookup_1/LookupTableFindV2 ^None_Lookup_2/LookupTableFindV2 ^None_Lookup_3/LookupTableFindV2 ^None_Lookup_4/LookupTableFindV2 ^None_Lookup_5/LookupTableFindV2 ^None_Lookup_6/LookupTableFindV2 ^None_Lookup_7/LookupTableFindV2^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 2>
None_Lookup/LookupTableFindV2None_Lookup/LookupTableFindV22B
None_Lookup_1/LookupTableFindV2None_Lookup_1/LookupTableFindV22B
None_Lookup_2/LookupTableFindV2None_Lookup_2/LookupTableFindV22B
None_Lookup_3/LookupTableFindV2None_Lookup_3/LookupTableFindV22B
None_Lookup_4/LookupTableFindV2None_Lookup_4/LookupTableFindV22B
None_Lookup_5/LookupTableFindV2None_Lookup_5/LookupTableFindV22B
None_Lookup_6/LookupTableFindV2None_Lookup_6/LookupTableFindV22B
None_Lookup_7/LookupTableFindV2None_Lookup_7/LookupTableFindV22
inference_opinference_op:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameage:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_gain:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_loss:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	education:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameeducation_num:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namefnlwgt:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namehours_per_week:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemarital_status:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namenative_country:O	K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
occupation:I
E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namerace:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerelationship:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namesex:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	workclass:,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_namemodel_handle

+
__inference__destroyer_2476
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 

Č
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2224
age	
capital_gain	
capital_loss	
	education
education_num	

fnlwgt	
hours_per_week	
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identity˘StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15**
Tin#
!2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *a
f\RZ
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2107o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameage:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_gain:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namecapital_loss:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	education:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameeducation_num:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namefnlwgt:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namehours_per_week:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemarital_status:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namenative_country:O	K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
occupation:I
E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namerace:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerelationship:HD
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namesex:NJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	workclass:$ 

_user_specified_name2188:

_output_shapes
: :$ 

_user_specified_name2192:

_output_shapes
: :$ 

_user_specified_name2196:

_output_shapes
: :$ 

_user_specified_name2200:

_output_shapes
: :$ 

_user_specified_name2204:

_output_shapes
: :$ 

_user_specified_name2208:

_output_shapes
: :$ 

_user_specified_name2212:

_output_shapes
: :$ 

_user_specified_name2216:

_output_shapes
: :$ 

_user_specified_name2220
Š
9
__inference__creator_2495
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1641*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
C
Î

__inference_call_2005

inputs	
	inputs_10	
	inputs_11	
inputs_3
inputs_4	
inputs_2	
	inputs_12	
inputs_5
	inputs_13
inputs_6
inputs_8
inputs_7
inputs_9
inputs_1.
*none_lookup_lookuptablefindv2_table_handle/
+none_lookup_lookuptablefindv2_default_value0
,none_lookup_1_lookuptablefindv2_table_handle1
-none_lookup_1_lookuptablefindv2_default_value0
,none_lookup_2_lookuptablefindv2_table_handle1
-none_lookup_2_lookuptablefindv2_default_value0
,none_lookup_3_lookuptablefindv2_table_handle1
-none_lookup_3_lookuptablefindv2_default_value0
,none_lookup_4_lookuptablefindv2_table_handle1
-none_lookup_4_lookuptablefindv2_default_value0
,none_lookup_5_lookuptablefindv2_table_handle1
-none_lookup_5_lookuptablefindv2_default_value0
,none_lookup_6_lookuptablefindv2_table_handle1
-none_lookup_6_lookuptablefindv2_default_value0
,none_lookup_7_lookuptablefindv2_table_handle1
-none_lookup_7_lookuptablefindv2_default_value
inference_op_model_handle
identity˘None_Lookup/LookupTableFindV2˘None_Lookup_1/LookupTableFindV2˘None_Lookup_2/LookupTableFindV2˘None_Lookup_3/LookupTableFindV2˘None_Lookup_4/LookupTableFindV2˘None_Lookup_5/LookupTableFindV2˘None_Lookup_6/LookupTableFindV2˘None_Lookup_7/LookupTableFindV2˘inference_opö
PartitionedCallPartitionedCallinputs	inputs_10	inputs_11inputs_3inputs_4inputs_2	inputs_12inputs_5	inputs_13inputs_6inputs_8inputs_7inputs_9inputs_1*
Tin
2						*
Tout
2*
_collective_manager_ids
 *č
_output_shapesŐ
Ň:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference__build_normalized_inputs_1946â
None_Lookup/LookupTableFindV2LookupTableFindV2*none_lookup_lookuptablefindv2_table_handlePartitionedCall:output:13+none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_1/LookupTableFindV2LookupTableFindV2,none_lookup_1_lookuptablefindv2_table_handlePartitionedCall:output:3-none_lookup_1_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_2/LookupTableFindV2LookupTableFindV2,none_lookup_2_lookuptablefindv2_table_handlePartitionedCall:output:7-none_lookup_2_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_3/LookupTableFindV2LookupTableFindV2,none_lookup_3_lookuptablefindv2_table_handlePartitionedCall:output:9-none_lookup_3_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_4/LookupTableFindV2LookupTableFindV2,none_lookup_4_lookuptablefindv2_table_handlePartitionedCall:output:11-none_lookup_4_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_5/LookupTableFindV2LookupTableFindV2,none_lookup_5_lookuptablefindv2_table_handlePartitionedCall:output:10-none_lookup_5_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_6/LookupTableFindV2LookupTableFindV2,none_lookup_6_lookuptablefindv2_table_handlePartitionedCall:output:12-none_lookup_6_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_7/LookupTableFindV2LookupTableFindV2,none_lookup_7_lookuptablefindv2_table_handlePartitionedCall:output:8-none_lookup_7_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  ¤
stack_1Pack(None_Lookup_1/LookupTableFindV2:values:0(None_Lookup_2/LookupTableFindV2:values:0(None_Lookup_7/LookupTableFindV2:values:0(None_Lookup_3/LookupTableFindV2:values:0(None_Lookup_5/LookupTableFindV2:values:0(None_Lookup_4/LookupTableFindV2:values:0(None_Lookup_6/LookupTableFindV2:values:0&None_Lookup/LookupTableFindV2:values:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisX
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0stack_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dim×
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference__finalize_predictions_2002i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ż
NoOpNoOp^None_Lookup/LookupTableFindV2 ^None_Lookup_1/LookupTableFindV2 ^None_Lookup_2/LookupTableFindV2 ^None_Lookup_3/LookupTableFindV2 ^None_Lookup_4/LookupTableFindV2 ^None_Lookup_5/LookupTableFindV2 ^None_Lookup_6/LookupTableFindV2 ^None_Lookup_7/LookupTableFindV2^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 2>
None_Lookup/LookupTableFindV2None_Lookup/LookupTableFindV22B
None_Lookup_1/LookupTableFindV2None_Lookup_1/LookupTableFindV22B
None_Lookup_2/LookupTableFindV2None_Lookup_2/LookupTableFindV22B
None_Lookup_3/LookupTableFindV2None_Lookup_3/LookupTableFindV22B
None_Lookup_4/LookupTableFindV2None_Lookup_4/LookupTableFindV22B
None_Lookup_5/LookupTableFindV2None_Lookup_5/LookupTableFindV22B
None_Lookup_6/LookupTableFindV2None_Lookup_6/LookupTableFindV22B
None_Lookup_7/LookupTableFindV2None_Lookup_7/LookupTableFindV22
inference_opinference_op:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_namemodel_handle

ö
__inference__initializer_25477
3key_value_init1658_lookuptableimportv2_table_handle/
+key_value_init1658_lookuptableimportv2_keys1
-key_value_init1658_lookuptableimportv2_values
identity˘&key_value_init1658/LookupTableImportV2ű
&key_value_init1658/LookupTableImportV2LookupTableImportV23key_value_init1658_lookuptableimportv2_table_handle+key_value_init1658_lookuptableimportv2_keys-key_value_init1658_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1658/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2P
&key_value_init1658/LookupTableImportV2&key_value_init1658/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
Š
9
__inference__creator_2510
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1647*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
F
Ĺ
__inference_call_2388

inputs_age	
inputs_capital_gain	
inputs_capital_loss	
inputs_education
inputs_education_num	
inputs_fnlwgt	
inputs_hours_per_week	
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclass.
*none_lookup_lookuptablefindv2_table_handle/
+none_lookup_lookuptablefindv2_default_value0
,none_lookup_1_lookuptablefindv2_table_handle1
-none_lookup_1_lookuptablefindv2_default_value0
,none_lookup_2_lookuptablefindv2_table_handle1
-none_lookup_2_lookuptablefindv2_default_value0
,none_lookup_3_lookuptablefindv2_table_handle1
-none_lookup_3_lookuptablefindv2_default_value0
,none_lookup_4_lookuptablefindv2_table_handle1
-none_lookup_4_lookuptablefindv2_default_value0
,none_lookup_5_lookuptablefindv2_table_handle1
-none_lookup_5_lookuptablefindv2_default_value0
,none_lookup_6_lookuptablefindv2_table_handle1
-none_lookup_6_lookuptablefindv2_default_value0
,none_lookup_7_lookuptablefindv2_table_handle1
-none_lookup_7_lookuptablefindv2_default_value
inference_op_model_handle
identity˘None_Lookup/LookupTableFindV2˘None_Lookup_1/LookupTableFindV2˘None_Lookup_2/LookupTableFindV2˘None_Lookup_3/LookupTableFindV2˘None_Lookup_4/LookupTableFindV2˘None_Lookup_5/LookupTableFindV2˘None_Lookup_6/LookupTableFindV2˘None_Lookup_7/LookupTableFindV2˘inference_opí
PartitionedCallPartitionedCall
inputs_ageinputs_capital_gaininputs_capital_lossinputs_educationinputs_education_numinputs_fnlwgtinputs_hours_per_weekinputs_marital_statusinputs_native_countryinputs_occupationinputs_raceinputs_relationship
inputs_sexinputs_workclass*
Tin
2						*
Tout
2*
_collective_manager_ids
 *č
_output_shapesŐ
Ň:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference__build_normalized_inputs_1946â
None_Lookup/LookupTableFindV2LookupTableFindV2*none_lookup_lookuptablefindv2_table_handlePartitionedCall:output:13+none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_1/LookupTableFindV2LookupTableFindV2,none_lookup_1_lookuptablefindv2_table_handlePartitionedCall:output:3-none_lookup_1_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_2/LookupTableFindV2LookupTableFindV2,none_lookup_2_lookuptablefindv2_table_handlePartitionedCall:output:7-none_lookup_2_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_3/LookupTableFindV2LookupTableFindV2,none_lookup_3_lookuptablefindv2_table_handlePartitionedCall:output:9-none_lookup_3_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_4/LookupTableFindV2LookupTableFindV2,none_lookup_4_lookuptablefindv2_table_handlePartitionedCall:output:11-none_lookup_4_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_5/LookupTableFindV2LookupTableFindV2,none_lookup_5_lookuptablefindv2_table_handlePartitionedCall:output:10-none_lookup_5_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙č
None_Lookup_6/LookupTableFindV2LookupTableFindV2,none_lookup_6_lookuptablefindv2_table_handlePartitionedCall:output:12-none_lookup_6_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙ç
None_Lookup_7/LookupTableFindV2LookupTableFindV2,none_lookup_7_lookuptablefindv2_table_handlePartitionedCall:output:8-none_lookup_7_lookuptablefindv2_default_value*	
Tin0*

Tout0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  ¤
stack_1Pack(None_Lookup_1/LookupTableFindV2:values:0(None_Lookup_2/LookupTableFindV2:values:0(None_Lookup_7/LookupTableFindV2:values:0(None_Lookup_3/LookupTableFindV2:values:0(None_Lookup_5/LookupTableFindV2:values:0(None_Lookup_4/LookupTableFindV2:values:0(None_Lookup_6/LookupTableFindV2:values:0&None_Lookup/LookupTableFindV2:values:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

axisX
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0stack_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dim×
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference__finalize_predictions_2002i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ż
NoOpNoOp^None_Lookup/LookupTableFindV2 ^None_Lookup_1/LookupTableFindV2 ^None_Lookup_2/LookupTableFindV2 ^None_Lookup_3/LookupTableFindV2 ^None_Lookup_4/LookupTableFindV2 ^None_Lookup_5/LookupTableFindV2 ^None_Lookup_6/LookupTableFindV2 ^None_Lookup_7/LookupTableFindV2^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes÷
ô:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : : : : : 2>
None_Lookup/LookupTableFindV2None_Lookup/LookupTableFindV22B
None_Lookup_1/LookupTableFindV2None_Lookup_1/LookupTableFindV22B
None_Lookup_2/LookupTableFindV2None_Lookup_2/LookupTableFindV22B
None_Lookup_3/LookupTableFindV2None_Lookup_3/LookupTableFindV22B
None_Lookup_4/LookupTableFindV2None_Lookup_4/LookupTableFindV22B
None_Lookup_5/LookupTableFindV2None_Lookup_5/LookupTableFindV22B
None_Lookup_6/LookupTableFindV2None_Lookup_6/LookupTableFindV22B
None_Lookup_7/LookupTableFindV2None_Lookup_7/LookupTableFindV22
inference_opinference_op:O K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_age:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_capital_gain:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_capital_loss:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_education:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_education_num:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinputs_fnlwgt:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_hours_per_week:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_marital_status:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_native_country:V	R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_occupation:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_race:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_relationship:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_sex:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_workclass:,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_nametable_handle:

_output_shapes
: :,(
&
_user_specified_namemodel_handle

+
__inference__destroyer_2521
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
"
Ę
)__inference__build_normalized_inputs_2314

inputs_age	
inputs_capital_gain	
inputs_capital_loss	
inputs_education
inputs_education_num	
inputs_fnlwgt	
inputs_hours_per_week	
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclass
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13U
CastCast
inputs_age*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Cast_1Castinputs_fnlwgt*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_2Castinputs_education_num*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Cast_3Castinputs_capital_gain*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Cast_4Castinputs_capital_loss*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙b
Cast_5Castinputs_hours_per_week*

DstT0*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙L
IdentityIdentityCast:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_1Identity
Cast_3:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_2Identity
Cast_4:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V

Identity_3Identityinputs_education*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_4Identity
Cast_2:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_5Identity
Cast_1:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_6Identity
Cast_5:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_7Identityinputs_marital_status*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_8Identityinputs_native_country*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W

Identity_9Identityinputs_occupation*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_10Identityinputs_race*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_11Identityinputs_relationship*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_12Identity
inputs_sex*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Identity_13Identityinputs_workclass*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*ç
_input_shapesŐ
Ň:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:O K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_age:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_capital_gain:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_capital_loss:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_education:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_education_num:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinputs_fnlwgt:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_hours_per_week:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_marital_status:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_native_country:V	R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_occupation:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_race:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_relationship:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs_sex:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_workclass
Š
9
__inference__creator_2465
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1629*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
Š
9
__inference__creator_2555
identity˘
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1665*
value_dtype0W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table

+
__inference__destroyer_2581
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 

ö
__inference__initializer_24877
3key_value_init1634_lookuptableimportv2_table_handle/
+key_value_init1634_lookuptableimportv2_keys1
-key_value_init1634_lookuptableimportv2_values
identity˘&key_value_init1634/LookupTableImportV2ű
&key_value_init1634/LookupTableImportV2LookupTableImportV23key_value_init1634_lookuptableimportv2_table_handle+key_value_init1634_lookuptableimportv2_keys-key_value_init1634_lookuptableimportv2_values*	
Tin0*

Tout0*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: K
NoOpNoOp'^key_value_init1634/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :	:	2P
&key_value_init1634/LookupTableImportV2&key_value_init1634/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:	: 

_output_shapes
:	
×$

__inference__traced_save_2697
file_prefix+
!read_disablecopyonread_is_trained:
 ,
"read_1_disablecopyonread_iteration:	 0
&read_2_disablecopyonread_learning_rate: 
savev2_const_24

identity_7˘MergeV2Checkpoints˘Read/DisableCopyOnRead˘Read/ReadVariableOp˘Read_1/DisableCopyOnRead˘Read_1/ReadVariableOp˘Read_2/DisableCopyOnRead˘Read_2/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_is_trained*
_output_shapes
 
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_is_trained^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0
R
IdentityIdentityRead/ReadVariableOp:value:0*
T0
*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0
*
_output_shapes
: g
Read_1/DisableCopyOnReadDisableCopyOnRead"read_1_disablecopyonread_iteration*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOp"read_1_disablecopyonread_iteration^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0	V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0	*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0	*
_output_shapes
: k
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_learning_rate*
_output_shapes
 
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_learning_rate^Read_2/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Â
value¸BľB&_is_trained/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHu
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0savev2_const_24"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_6Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_7IdentityIdentity_6:output:0^NoOp*
T0*
_output_shapes
: Ě
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_user_specified_name
is_trained:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:@<

_output_shapes
: 
"
_user_specified_name
Const_24
ô
Z
&__inference__finalize_predictions_2002
predictions
predictions_1
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      é
strided_sliceStridedSlicepredictionsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

begin_mask*
end_mask^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:˙˙˙˙˙˙˙˙˙::T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepredictions:GC

_output_shapes
:
%
_user_specified_namepredictions"ŹN
saver_filename:0StatefulPartitionedCall_10:0StatefulPartitionedCall_118"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ö
serving_defaultÂ
/
age(
serving_default_age:0	˙˙˙˙˙˙˙˙˙
A
capital_gain1
serving_default_capital_gain:0	˙˙˙˙˙˙˙˙˙
A
capital_loss1
serving_default_capital_loss:0	˙˙˙˙˙˙˙˙˙
;
	education.
serving_default_education:0˙˙˙˙˙˙˙˙˙
C
education_num2
serving_default_education_num:0	˙˙˙˙˙˙˙˙˙
5
fnlwgt+
serving_default_fnlwgt:0	˙˙˙˙˙˙˙˙˙
E
hours_per_week3
 serving_default_hours_per_week:0	˙˙˙˙˙˙˙˙˙
E
marital_status3
 serving_default_marital_status:0˙˙˙˙˙˙˙˙˙
E
native_country3
 serving_default_native_country:0˙˙˙˙˙˙˙˙˙
=

occupation/
serving_default_occupation:0˙˙˙˙˙˙˙˙˙
1
race)
serving_default_race:0˙˙˙˙˙˙˙˙˙
A
relationship1
serving_default_relationship:0˙˙˙˙˙˙˙˙˙
/
sex(
serving_default_sex:0˙˙˙˙˙˙˙˙˙
;
	workclass.
serving_default_workclass:0˙˙˙˙˙˙˙˙˙<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict22

asset_path_initializer:03bb60e6d9306498ddone2D

asset_path_initializer_1:0$3bb60e6d9306498dnodes-00000-of-0000129

asset_path_initializer_2:03bb60e6d9306498dheader.pb2P

asset_path_initializer_3:003bb60e6d9306498dgradient_boosted_trees_header.pb2<

asset_path_initializer_4:03bb60e6d9306498ddata_spec.pb:ţ
ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

_multitask
	_is_trained

_learner_params
	_features
	optimizer
loss
_models
_build_normalized_inputs
_finalize_predictions
call
call_get_leaves
yggdrasil_model_path_tensor

signatures"
_tf_keras_model
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
á
trace_0
trace_12Ş
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2224
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2276Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1

trace_0
trace_12ŕ
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2107
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2172Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
Ú
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15Bá
__inference__wrapped_model_2042agecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass"
˛
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z	capture_1z	capture_3z 	capture_5z!	capture_7z"	capture_9z#
capture_11z$
capture_13z%
capture_15
 "
trackable_list_wrapper
:
 2
is_trained
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
j
&
_variables
'_iterations
(_learning_rate
)_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
'
*0"
trackable_list_wrapper
ă
+trace_02Ć
)__inference__build_normalized_inputs_2314
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z+trace_0

,trace_02ä
&__inference__finalize_predictions_2323š
˛˛Ž
FullArgSpec1
args)&
jtask
jpredictions
jlike_engine
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z,trace_0
ŕ
-trace_02Ă
__inference_call_2388Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z-trace_0
2
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ű
.trace_02Ţ
,__inference_yggdrasil_model_path_tensor_2393­
Ľ˛Ą
FullArgSpec$
args
jmultitask_model_index
varargs
 
varkw
 
defaults˘
` 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z.trace_0
,
/serving_default"
signature_map
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15B
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2224agecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z	capture_1z	capture_3z 	capture_5z!	capture_7z"	capture_9z#
capture_11z$
capture_13z%
capture_15

	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15B
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2276agecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z	capture_1z	capture_3z 	capture_5z!	capture_7z"	capture_9z#
capture_11z$
capture_13z%
capture_15

	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15BŚ
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2107agecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z	capture_1z	capture_3z 	capture_5z!	capture_7z"	capture_9z#
capture_11z$
capture_13z%
capture_15

	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15BŚ
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2172agecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z	capture_1z	capture_3z 	capture_5z!	capture_7z"	capture_9z#
capture_11z$
capture_13z%
capture_15
"J

Const_17jtf.TrackableConstant
"J

Const_16jtf.TrackableConstant
"J

Const_23jtf.TrackableConstant
"J

Const_22jtf.TrackableConstant
"J

Const_21jtf.TrackableConstant
"J

Const_20jtf.TrackableConstant
"J

Const_19jtf.TrackableConstant
"J

Const_18jtf.TrackableConstant
'
'0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
ľ2˛Ż
Ś˛˘
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 0
G
0_input_builder
1_compiled_model"
_generic_user_object
ĐBÍ
)__inference__build_normalized_inputs_2314
inputs_ageinputs_capital_gaininputs_capital_lossinputs_educationinputs_education_numinputs_fnlwgtinputs_hours_per_weekinputs_marital_statusinputs_native_countryinputs_occupationinputs_raceinputs_relationship
inputs_sexinputs_workclass"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŠBŚ
&__inference__finalize_predictions_2323predictions_dense_predictions$predictions_dense_col_representation"´
­˛Š
FullArgSpec1
args)&
jtask
jpredictions
jlike_engine
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ž
	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15BĹ
__inference_call_2388
inputs_ageinputs_capital_gaininputs_capital_lossinputs_educationinputs_education_numinputs_fnlwgtinputs_hours_per_weekinputs_marital_statusinputs_native_countryinputs_occupationinputs_raceinputs_relationship
inputs_sexinputs_workclass"¤
˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z	capture_1z	capture_3z 	capture_5z!	capture_7z"	capture_9z#
capture_11z$
capture_13z%
capture_15
ů
2	capture_0BŘ
,__inference_yggdrasil_model_path_tensor_2393"§
 ˛
FullArgSpec$
args
jmultitask_model_index
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z2	capture_0

	capture_1
	capture_3
 	capture_5
!	capture_7
"	capture_9
#
capture_11
$
capture_13
%
capture_15B
"__inference_signature_wrapper_2446agecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclass"Đ
É˛Ĺ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 Ň

kwonlyargsĂż
jage
jcapital_gain
jcapital_loss
j	education
jeducation_num
jfnlwgt
jhours_per_week
jmarital_status
jnative_country
j
occupation
jrace
jrelationship
jsex
j	workclass
kwonlydefaults
 
annotationsŞ *
 z	capture_1z	capture_3z 	capture_5z!	capture_7z"	capture_9z#
capture_11z$
capture_13z%
capture_15
l
3_feature_name_to_idx
4	_init_ops
#5categorical_str_to_int_hashmaps"
_generic_user_object
S
6_model_loader
7_create_resource
8_initialize
9_destroy_resourceR 
* 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

:	education
;marital_status
<native_country
=
occupation
>race
?relationship
@sex
A	workclass"
trackable_dict_wrapper
Q
B_output_types
C
_all_files
2
_done_file"
_generic_user_object
Ę
Dtrace_02­
__inference__creator_2450
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zDtrace_0
Î
Etrace_02ą
__inference__initializer_2457
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zEtrace_0
Ě
Ftrace_02Ż
__inference__destroyer_2461
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zFtrace_0
f
G_initializer
H_create_resource
I_initialize
J_destroy_resourceR jtf.StaticHashTable
f
K_initializer
L_create_resource
M_initialize
N_destroy_resourceR jtf.StaticHashTable
f
O_initializer
P_create_resource
Q_initialize
R_destroy_resourceR jtf.StaticHashTable
f
S_initializer
T_create_resource
U_initialize
V_destroy_resourceR jtf.StaticHashTable
f
W_initializer
X_create_resource
Y_initialize
Z_destroy_resourceR jtf.StaticHashTable
f
[_initializer
\_create_resource
]_initialize
^_destroy_resourceR jtf.StaticHashTable
f
__initializer
`_create_resource
a_initialize
b_destroy_resourceR jtf.StaticHashTable
f
c_initializer
d_create_resource
e_initialize
f_destroy_resourceR jtf.StaticHashTable
 "
trackable_list_wrapper
C
g0
21
h2
i3
j4"
trackable_list_wrapper
°B­
__inference__creator_2450"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
Ň
2	capture_0Bą
__inference__initializer_2457"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z2	capture_0
˛BŻ
__inference__destroyer_2461"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
"
_generic_user_object
Ę
ktrace_02­
__inference__creator_2465
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zktrace_0
Î
ltrace_02ą
__inference__initializer_2472
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zltrace_0
Ě
mtrace_02Ż
__inference__destroyer_2476
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zmtrace_0
"
_generic_user_object
Ę
ntrace_02­
__inference__creator_2480
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zntrace_0
Î
otrace_02ą
__inference__initializer_2487
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zotrace_0
Ě
ptrace_02Ż
__inference__destroyer_2491
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zptrace_0
"
_generic_user_object
Ę
qtrace_02­
__inference__creator_2495
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zqtrace_0
Î
rtrace_02ą
__inference__initializer_2502
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zrtrace_0
Ě
strace_02Ż
__inference__destroyer_2506
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zstrace_0
"
_generic_user_object
Ę
ttrace_02­
__inference__creator_2510
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zttrace_0
Î
utrace_02ą
__inference__initializer_2517
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zutrace_0
Ě
vtrace_02Ż
__inference__destroyer_2521
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zvtrace_0
"
_generic_user_object
Ę
wtrace_02­
__inference__creator_2525
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zwtrace_0
Î
xtrace_02ą
__inference__initializer_2532
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zxtrace_0
Ě
ytrace_02Ż
__inference__destroyer_2536
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zytrace_0
"
_generic_user_object
Ę
ztrace_02­
__inference__creator_2540
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zztrace_0
Î
{trace_02ą
__inference__initializer_2547
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z{trace_0
Ě
|trace_02Ż
__inference__destroyer_2551
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z|trace_0
"
_generic_user_object
Ę
}trace_02­
__inference__creator_2555
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z}trace_0
Î
~trace_02ą
__inference__initializer_2562
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z~trace_0
Ě
trace_02Ż
__inference__destroyer_2566
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ ztrace_0
"
_generic_user_object
Ě
trace_02­
__inference__creator_2570
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ ztrace_0
Đ
trace_02ą
__inference__initializer_2577
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ ztrace_0
Î
trace_02Ż
__inference__destroyer_2581
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ ztrace_0
*
*
*
*
°B­
__inference__creator_2465"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2472"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2476"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
°B­
__inference__creator_2480"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2487"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2491"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
°B­
__inference__creator_2495"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2502"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2506"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
°B­
__inference__creator_2510"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2517"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2521"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
°B­
__inference__creator_2525"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2532"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2536"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
°B­
__inference__creator_2540"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2547"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2551"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
°B­
__inference__creator_2555"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2562"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2566"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
°B­
__inference__creator_2570"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ô
	capture_1
	capture_2Bą
__inference__initializer_2577"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
˛BŻ
__inference__destroyer_2581"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
"J

Const_15jtf.TrackableConstant
"J

Const_14jtf.TrackableConstant
"J

Const_13jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstantÝ
)__inference__build_normalized_inputs_2314Ż˘
˘˙
üŞř
'
age 

inputs_age˙˙˙˙˙˙˙˙˙	
9
capital_gain)&
inputs_capital_gain˙˙˙˙˙˙˙˙˙	
9
capital_loss)&
inputs_capital_loss˙˙˙˙˙˙˙˙˙	
3
	education&#
inputs_education˙˙˙˙˙˙˙˙˙
;
education_num*'
inputs_education_num˙˙˙˙˙˙˙˙˙	
-
fnlwgt# 
inputs_fnlwgt˙˙˙˙˙˙˙˙˙	
=
hours_per_week+(
inputs_hours_per_week˙˙˙˙˙˙˙˙˙	
=
marital_status+(
inputs_marital_status˙˙˙˙˙˙˙˙˙
=
native_country+(
inputs_native_country˙˙˙˙˙˙˙˙˙
5

occupation'$
inputs_occupation˙˙˙˙˙˙˙˙˙
)
race!
inputs_race˙˙˙˙˙˙˙˙˙
9
relationship)&
inputs_relationship˙˙˙˙˙˙˙˙˙
'
sex 

inputs_sex˙˙˙˙˙˙˙˙˙
3
	workclass&#
inputs_workclass˙˙˙˙˙˙˙˙˙
Ş "Ş
 
age
age˙˙˙˙˙˙˙˙˙
2
capital_gain"
capital_gain˙˙˙˙˙˙˙˙˙
2
capital_loss"
capital_loss˙˙˙˙˙˙˙˙˙
,
	education
	education˙˙˙˙˙˙˙˙˙
4
education_num# 
education_num˙˙˙˙˙˙˙˙˙
&
fnlwgt
fnlwgt˙˙˙˙˙˙˙˙˙
6
hours_per_week$!
hours_per_week˙˙˙˙˙˙˙˙˙
6
marital_status$!
marital_status˙˙˙˙˙˙˙˙˙
6
native_country$!
native_country˙˙˙˙˙˙˙˙˙
.

occupation 

occupation˙˙˙˙˙˙˙˙˙
"
race
race˙˙˙˙˙˙˙˙˙
2
relationship"
relationship˙˙˙˙˙˙˙˙˙
 
sex
sex˙˙˙˙˙˙˙˙˙
,
	workclass
	workclass˙˙˙˙˙˙˙˙˙>
__inference__creator_2450!˘

˘ 
Ş "
unknown >
__inference__creator_2465!˘

˘ 
Ş "
unknown >
__inference__creator_2480!˘

˘ 
Ş "
unknown >
__inference__creator_2495!˘

˘ 
Ş "
unknown >
__inference__creator_2510!˘

˘ 
Ş "
unknown >
__inference__creator_2525!˘

˘ 
Ş "
unknown >
__inference__creator_2540!˘

˘ 
Ş "
unknown >
__inference__creator_2555!˘

˘ 
Ş "
unknown >
__inference__creator_2570!˘

˘ 
Ş "
unknown @
__inference__destroyer_2461!˘

˘ 
Ş "
unknown @
__inference__destroyer_2476!˘

˘ 
Ş "
unknown @
__inference__destroyer_2491!˘

˘ 
Ş "
unknown @
__inference__destroyer_2506!˘

˘ 
Ş "
unknown @
__inference__destroyer_2521!˘

˘ 
Ş "
unknown @
__inference__destroyer_2536!˘

˘ 
Ş "
unknown @
__inference__destroyer_2551!˘

˘ 
Ş "
unknown @
__inference__destroyer_2566!˘

˘ 
Ş "
unknown @
__inference__destroyer_2581!˘

˘ 
Ş "
unknown 
&__inference__finalize_predictions_2323ďÉ˘Ĺ
˝˘š
`
Ž˛Ş
ModelOutputL
dense_predictions74
predictions_dense_predictions˙˙˙˙˙˙˙˙˙M
dense_col_representation1.
$predictions_dense_col_representation
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙F
__inference__initializer_2457%21˘

˘ 
Ş "
unknown I
__inference__initializer_2472(:˘

˘ 
Ş "
unknown I
__inference__initializer_2487(;˘

˘ 
Ş "
unknown I
__inference__initializer_2502(<˘

˘ 
Ş "
unknown I
__inference__initializer_2517(=˘

˘ 
Ş "
unknown I
__inference__initializer_2532(>˘

˘ 
Ş "
unknown I
__inference__initializer_2547(?˘

˘ 
Ş "
unknown I
__inference__initializer_2562(@˘

˘ 
Ş "
unknown I
__inference__initializer_2577(A˘

˘ 
Ş "
unknown 
__inference__wrapped_model_2042řA:; =!?">#@$<%1­˘Š
Ą˘
Ş
 
age
age˙˙˙˙˙˙˙˙˙	
2
capital_gain"
capital_gain˙˙˙˙˙˙˙˙˙	
2
capital_loss"
capital_loss˙˙˙˙˙˙˙˙˙	
,
	education
	education˙˙˙˙˙˙˙˙˙
4
education_num# 
education_num˙˙˙˙˙˙˙˙˙	
&
fnlwgt
fnlwgt˙˙˙˙˙˙˙˙˙	
6
hours_per_week$!
hours_per_week˙˙˙˙˙˙˙˙˙	
6
marital_status$!
marital_status˙˙˙˙˙˙˙˙˙
6
native_country$!
native_country˙˙˙˙˙˙˙˙˙
.

occupation 

occupation˙˙˙˙˙˙˙˙˙
"
race
race˙˙˙˙˙˙˙˙˙
2
relationship"
relationship˙˙˙˙˙˙˙˙˙
 
sex
sex˙˙˙˙˙˙˙˙˙
,
	workclass
	workclass˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙ć
__inference_call_2388ĚA:; =!?">#@$<%1˘
˘
üŞř
'
age 

inputs_age˙˙˙˙˙˙˙˙˙	
9
capital_gain)&
inputs_capital_gain˙˙˙˙˙˙˙˙˙	
9
capital_loss)&
inputs_capital_loss˙˙˙˙˙˙˙˙˙	
3
	education&#
inputs_education˙˙˙˙˙˙˙˙˙
;
education_num*'
inputs_education_num˙˙˙˙˙˙˙˙˙	
-
fnlwgt# 
inputs_fnlwgt˙˙˙˙˙˙˙˙˙	
=
hours_per_week+(
inputs_hours_per_week˙˙˙˙˙˙˙˙˙	
=
marital_status+(
inputs_marital_status˙˙˙˙˙˙˙˙˙
=
native_country+(
inputs_native_country˙˙˙˙˙˙˙˙˙
5

occupation'$
inputs_occupation˙˙˙˙˙˙˙˙˙
)
race!
inputs_race˙˙˙˙˙˙˙˙˙
9
relationship)&
inputs_relationship˙˙˙˙˙˙˙˙˙
'
sex 

inputs_sex˙˙˙˙˙˙˙˙˙
3
	workclass&#
inputs_workclass˙˙˙˙˙˙˙˙˙
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙Ň
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2107őA:; =!?">#@$<%1ą˘­
Ľ˘Ą
Ş
 
age
age˙˙˙˙˙˙˙˙˙	
2
capital_gain"
capital_gain˙˙˙˙˙˙˙˙˙	
2
capital_loss"
capital_loss˙˙˙˙˙˙˙˙˙	
,
	education
	education˙˙˙˙˙˙˙˙˙
4
education_num# 
education_num˙˙˙˙˙˙˙˙˙	
&
fnlwgt
fnlwgt˙˙˙˙˙˙˙˙˙	
6
hours_per_week$!
hours_per_week˙˙˙˙˙˙˙˙˙	
6
marital_status$!
marital_status˙˙˙˙˙˙˙˙˙
6
native_country$!
native_country˙˙˙˙˙˙˙˙˙
.

occupation 

occupation˙˙˙˙˙˙˙˙˙
"
race
race˙˙˙˙˙˙˙˙˙
2
relationship"
relationship˙˙˙˙˙˙˙˙˙
 
sex
sex˙˙˙˙˙˙˙˙˙
,
	workclass
	workclass˙˙˙˙˙˙˙˙˙
p
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Ň
X__inference_gradient_boosted_trees_model_1_layer_call_and_return_conditional_losses_2172őA:; =!?">#@$<%1ą˘­
Ľ˘Ą
Ş
 
age
age˙˙˙˙˙˙˙˙˙	
2
capital_gain"
capital_gain˙˙˙˙˙˙˙˙˙	
2
capital_loss"
capital_loss˙˙˙˙˙˙˙˙˙	
,
	education
	education˙˙˙˙˙˙˙˙˙
4
education_num# 
education_num˙˙˙˙˙˙˙˙˙	
&
fnlwgt
fnlwgt˙˙˙˙˙˙˙˙˙	
6
hours_per_week$!
hours_per_week˙˙˙˙˙˙˙˙˙	
6
marital_status$!
marital_status˙˙˙˙˙˙˙˙˙
6
native_country$!
native_country˙˙˙˙˙˙˙˙˙
.

occupation 

occupation˙˙˙˙˙˙˙˙˙
"
race
race˙˙˙˙˙˙˙˙˙
2
relationship"
relationship˙˙˙˙˙˙˙˙˙
 
sex
sex˙˙˙˙˙˙˙˙˙
,
	workclass
	workclass˙˙˙˙˙˙˙˙˙
p 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 Ź
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2224ęA:; =!?">#@$<%1ą˘­
Ľ˘Ą
Ş
 
age
age˙˙˙˙˙˙˙˙˙	
2
capital_gain"
capital_gain˙˙˙˙˙˙˙˙˙	
2
capital_loss"
capital_loss˙˙˙˙˙˙˙˙˙	
,
	education
	education˙˙˙˙˙˙˙˙˙
4
education_num# 
education_num˙˙˙˙˙˙˙˙˙	
&
fnlwgt
fnlwgt˙˙˙˙˙˙˙˙˙	
6
hours_per_week$!
hours_per_week˙˙˙˙˙˙˙˙˙	
6
marital_status$!
marital_status˙˙˙˙˙˙˙˙˙
6
native_country$!
native_country˙˙˙˙˙˙˙˙˙
.

occupation 

occupation˙˙˙˙˙˙˙˙˙
"
race
race˙˙˙˙˙˙˙˙˙
2
relationship"
relationship˙˙˙˙˙˙˙˙˙
 
sex
sex˙˙˙˙˙˙˙˙˙
,
	workclass
	workclass˙˙˙˙˙˙˙˙˙
p
Ş "!
unknown˙˙˙˙˙˙˙˙˙Ź
=__inference_gradient_boosted_trees_model_1_layer_call_fn_2276ęA:; =!?">#@$<%1ą˘­
Ľ˘Ą
Ş
 
age
age˙˙˙˙˙˙˙˙˙	
2
capital_gain"
capital_gain˙˙˙˙˙˙˙˙˙	
2
capital_loss"
capital_loss˙˙˙˙˙˙˙˙˙	
,
	education
	education˙˙˙˙˙˙˙˙˙
4
education_num# 
education_num˙˙˙˙˙˙˙˙˙	
&
fnlwgt
fnlwgt˙˙˙˙˙˙˙˙˙	
6
hours_per_week$!
hours_per_week˙˙˙˙˙˙˙˙˙	
6
marital_status$!
marital_status˙˙˙˙˙˙˙˙˙
6
native_country$!
native_country˙˙˙˙˙˙˙˙˙
.

occupation 

occupation˙˙˙˙˙˙˙˙˙
"
race
race˙˙˙˙˙˙˙˙˙
2
relationship"
relationship˙˙˙˙˙˙˙˙˙
 
sex
sex˙˙˙˙˙˙˙˙˙
,
	workclass
	workclass˙˙˙˙˙˙˙˙˙
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙
"__inference_signature_wrapper_2446ńA:; =!?">#@$<%1Ś˘˘
˘ 
Ş
 
age
age˙˙˙˙˙˙˙˙˙	
2
capital_gain"
capital_gain˙˙˙˙˙˙˙˙˙	
2
capital_loss"
capital_loss˙˙˙˙˙˙˙˙˙	
,
	education
	education˙˙˙˙˙˙˙˙˙
4
education_num# 
education_num˙˙˙˙˙˙˙˙˙	
&
fnlwgt
fnlwgt˙˙˙˙˙˙˙˙˙	
6
hours_per_week$!
hours_per_week˙˙˙˙˙˙˙˙˙	
6
marital_status$!
marital_status˙˙˙˙˙˙˙˙˙
6
native_country$!
native_country˙˙˙˙˙˙˙˙˙
.

occupation 

occupation˙˙˙˙˙˙˙˙˙
"
race
race˙˙˙˙˙˙˙˙˙
2
relationship"
relationship˙˙˙˙˙˙˙˙˙
 
sex
sex˙˙˙˙˙˙˙˙˙
,
	workclass
	workclass˙˙˙˙˙˙˙˙˙"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙X
,__inference_yggdrasil_model_path_tensor_2393(2˘
˘
` 
Ş "
unknown 