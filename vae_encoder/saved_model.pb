Щў
єМ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8тс	
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0
}
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*!
shared_namez_log_var/kernel
v
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes
:	А*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
w
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namez_mean/kernel
p
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes
:	А*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:А*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
АА*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	А*
dtype0
Ф
embedding_MARITAL/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameembedding_MARITAL/embeddings
Н
0embedding_MARITAL/embeddings/Read/ReadVariableOpReadVariableOpembedding_MARITAL/embeddings*
_output_shapes

:*
dtype0
Ш
embedding_ETHNICITY/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name embedding_ETHNICITY/embeddings
С
2embedding_ETHNICITY/embeddings/Read/ReadVariableOpReadVariableOpembedding_ETHNICITY/embeddings*
_output_shapes

:*
dtype0
О
embedding_RACE/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameembedding_RACE/embeddings
З
-embedding_RACE/embeddings/Read/ReadVariableOpReadVariableOpembedding_RACE/embeddings*
_output_shapes

:*
dtype0
Т
embedding_GENDER/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameembedding_GENDER/embeddings
Л
/embedding_GENDER/embeddings/Read/ReadVariableOpReadVariableOpembedding_GENDER/embeddings*
_output_shapes

:*
dtype0
В
serving_default_input_ETHNICITYPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€

serving_default_input_GENDERPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
А
serving_default_input_MARITALPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
}
serving_default_input_RACEPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Г
 serving_default_input_continuousPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
з
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_ETHNICITYserving_default_input_GENDERserving_default_input_MARITALserving_default_input_RACE serving_default_input_continuousembedding_MARITAL/embeddingsembedding_ETHNICITY/embeddingsembedding_RACE/embeddingsembedding_GENDER/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_83550

NoOpNoOp
ШP
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*”O
value…OB∆O BњO
Н
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer_with_weights-5
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
* 
* 
†
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"
embeddings*
†
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
embeddings*
†
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0
embeddings*
†
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7
embeddings*
О
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
О
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
О
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 
О
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
* 
О
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
¶
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*
¶
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias*
¶
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias*
¶
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias*
О
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
Z
"0
)1
02
73
\4
]5
d6
e7
l8
m9
t10
u11*
Z
"0
)1
02
73
\4
]5
d6
e7
l8
m9
t10
u11*
* 
±
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Бtrace_0
Вtrace_1
Гtrace_2
Дtrace_3* 
:
Еtrace_0
Жtrace_1
Зtrace_2
Иtrace_3* 
* 

Йserving_default* 

"0*

"0*
* 
Ш
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Пtrace_0* 

Рtrace_0* 
oi
VARIABLE_VALUEembedding_GENDER/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

)0*

)0*
* 
Ш
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Цtrace_0* 

Чtrace_0* 
mg
VARIABLE_VALUEembedding_RACE/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

00*

00*
* 
Ш
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Эtrace_0* 

Юtrace_0* 
rl
VARIABLE_VALUEembedding_ETHNICITY/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

70*

70*
* 
Ш
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

§trace_0* 

•trace_0* 
pj
VARIABLE_VALUEembedding_MARITAL/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

Ђtrace_0* 

ђtrace_0* 
* 
* 
* 
Ц
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

≤trace_0* 

≥trace_0* 
* 
* 
* 
Ц
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

єtrace_0* 

Їtrace_0* 
* 
* 
* 
Ц
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

јtrace_0* 

Ѕtrace_0* 
* 
* 
* 
Ц
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

«trace_0* 

»trace_0* 

\0
]1*

\0
]1*
* 
Ш
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

ќtrace_0* 

ѕtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

d0
e1*
* 
Ш
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

’trace_0* 

÷trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

l0
m1*

l0
m1*
* 
Ш
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

№trace_0* 

Ёtrace_0* 
]W
VARIABLE_VALUEz_mean/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEz_mean/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

t0
u1*

t0
u1*
* 
Ш
ёnon_trainable_variables
яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

гtrace_0* 

дtrace_0* 
`Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEz_log_var/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

кtrace_0
лtrace_1* 

мtrace_0
нtrace_1* 
* 
Т
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18*
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
Е
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/embedding_GENDER/embeddings/Read/ReadVariableOp-embedding_RACE/embeddings/Read/ReadVariableOp2embedding_ETHNICITY/embeddings/Read/ReadVariableOp0embedding_MARITAL/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_84128
Р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_GENDER/embeddingsembedding_RACE/embeddingsembedding_ETHNICITY/embeddingsembedding_MARITAL/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_84174Шф
»	
у
A__inference_z_mean_layer_call_and_return_conditional_losses_83988

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_82977

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
§
…
'__inference_encoder_layer_call_fn_83587
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	А
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:
identity

identity_1

identity_2ИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputs_input_ethnicityinputs_input_genderinputs_input_maritalinputs_input_raceinputs_input_continuousunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_83084o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs/input_continuous
®
E
)__inference_flatten_1_layer_call_fn_83882

inputs
identity≤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_82961`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
µ
Е
1__inference_embedding_MARITAL_layer_call_fn_83856

inputs
unknown:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_82901s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Я

у
@__inference_dense_layer_call_and_return_conditional_losses_83002

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°	
•
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_82929

inputs(
embedding_lookup_82923:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_82923Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/82923*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/82923*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶	
™
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_82915

inputs(
embedding_lookup_82909:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_82909Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/82909*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/82909*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»	
у
A__inference_z_mean_layer_call_and_return_conditional_losses_83035

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_83910

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≥
Д
0__inference_embedding_GENDER_layer_call_fn_83805

inputs
unknown:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_82943s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®
E
)__inference_flatten_3_layer_call_fn_83904

inputs
identity≤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_82977`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ї
¶
'__inference_encoder_layer_call_fn_83115
input_ethnicity
input_gender
input_marital

input_race
input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	А
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:
identity

identity_1

identity_2ИҐStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityinput_genderinput_marital
input_raceinput_continuousunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_83084o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinput_continuous
Љ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_83888

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
—3
г
!__inference__traced_restore_84174
file_prefix>
,assignvariableop_embedding_gender_embeddings:>
,assignvariableop_1_embedding_race_embeddings:C
1assignvariableop_2_embedding_ethnicity_embeddings:A
/assignvariableop_3_embedding_marital_embeddings:2
assignvariableop_4_dense_kernel:	А,
assignvariableop_5_dense_bias:	А5
!assignvariableop_6_dense_1_kernel:
АА.
assignvariableop_7_dense_1_bias:	А3
 assignvariableop_8_z_mean_kernel:	А,
assignvariableop_9_z_mean_bias:7
$assignvariableop_10_z_log_var_kernel:	А0
"assignvariableop_11_z_log_var_bias:
identity_13ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9µ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ
value—BќB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHК
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B я
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOpAssignVariableOp,assignvariableop_embedding_gender_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_1AssignVariableOp,assignvariableop_1_embedding_race_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_2AssignVariableOp1assignvariableop_2_embedding_ethnicity_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_3AssignVariableOp/assignvariableop_3_embedding_marital_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_8AssignVariableOp assignvariableop_8_z_mean_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_z_mean_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_10AssignVariableOp$assignvariableop_10_z_log_var_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_11AssignVariableOp"assignvariableop_11_z_log_var_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 „
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: ƒ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°	
•
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_83832

inputs(
embedding_lookup_83826:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_83826Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/83826*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/83826*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ї
^
B__inference_flatten_layer_call_and_return_conditional_losses_83877

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•

ц
B__inference_dense_1_layer_call_and_return_conditional_losses_83019

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
є
З
3__inference_embedding_ETHNICITY_layer_call_fn_83839

inputs
unknown:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_82915s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ћ	
ц
D__inference_z_log_var_layer_call_and_return_conditional_losses_83051

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
М
i
<__inference_z_layer_call_and_return_conditional_losses_83079

inputs
inputs_1
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≥
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0*

seed**
seed2БУћЦ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€E
ExpExpmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠C
Ќ
B__inference_encoder_layer_call_and_return_conditional_losses_83463
input_ethnicity
input_gender
input_marital

input_race
input_continuous)
embedding_marital_83422:+
embedding_ethnicity_83425:&
embedding_race_83428:(
embedding_gender_83431:
dense_83439:	А
dense_83441:	А!
dense_1_83444:
АА
dense_1_83446:	А
z_mean_83449:	А
z_mean_83451:"
z_log_var_83454:	А
z_log_var_83456:
identity

identity_1

identity_2ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ+embedding_ETHNICITY/StatefulPartitionedCallҐ(embedding_GENDER/StatefulPartitionedCallҐ)embedding_MARITAL/StatefulPartitionedCallҐ&embedding_RACE/StatefulPartitionedCallҐz/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCallД
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinput_maritalembedding_marital_83422*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_82901М
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityembedding_ethnicity_83425*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_82915ш
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCall
input_raceembedding_race_83428*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_82929А
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinput_genderembedding_gender_83431*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_82943г
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_82953е
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_82961к
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_82969и
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_82977№
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0input_continuous*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_82989Г
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_83439dense_83441*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83002Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_83444dense_1_83446*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83019К
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_83449z_mean_83451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_83035Ц
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_83454z_log_var_83456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_83051К
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_83079v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ч
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+embedding_ETHNICITY/StatefulPartitionedCall+embedding_ETHNICITY/StatefulPartitionedCall2T
(embedding_GENDER/StatefulPartitionedCall(embedding_GENDER/StatefulPartitionedCall2V
)embedding_MARITAL/StatefulPartitionedCall)embedding_MARITAL/StatefulPartitionedCall2P
&embedding_RACE/StatefulPartitionedCall&embedding_RACE/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinput_continuous
®
E
)__inference_flatten_2_layer_call_fn_83893

inputs
identity≤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_82969`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ї
Ь
F__inference_concatenate_layer_call_and_return_conditional_losses_83929
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Х
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4
£	
І
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_82943

inputs(
embedding_lookup_82937:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_82937Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/82937*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/82937*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ц
k
<__inference_z_layer_call_and_return_conditional_losses_84063
inputs_0
inputs_1
identityИ=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≥
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0*

seed**
seed2ХаўЦ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€E
ExpExpmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
£	
І
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_83815

inputs(
embedding_lookup_83809:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_83809Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/83809*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/83809*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
§
…
'__inference_encoder_layer_call_fn_83624
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	А
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:
identity

identity_1

identity_2ИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputs_input_ethnicityinputs_input_genderinputs_input_maritalinputs_input_raceinputs_input_continuousunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_83347o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs/input_continuous
¬
Ф
&__inference_z_mean_layer_call_fn_83978

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_83035o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_82969

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
§	
®
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_83866

inputs(
embedding_lookup_83860:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_83860Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/83860*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/83860*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠
Ъ
F__inference_concatenate_layer_call_and_return_conditional_losses_82989

inputs
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :У
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Х
Ґ
#__inference_signature_wrapper_83550
input_ethnicity
input_gender
input_marital

input_race
input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	А
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:
identity

identity_1

identity_2ИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityinput_genderinput_marital
input_raceinput_continuousunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_82876o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinput_continuous
Љ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_82961

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•

ц
B__inference_dense_1_layer_call_and_return_conditional_losses_83969

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ц
k
<__inference_z_layer_call_and_return_conditional_losses_84041
inputs_0
inputs_1
identityИ=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≥
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0*

seed**
seed2єЪ’Ц
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€E
ExpExpmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
ї
¶
'__inference_encoder_layer_call_fn_83415
input_ethnicity
input_gender
input_marital

input_race
input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	А
	unknown_4:	А
	unknown_5:
АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:
	unknown_9:	А

unknown_10:
identity

identity_1

identity_2ИҐStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityinput_genderinput_marital
input_raceinput_continuousunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_83347o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinput_continuous
§	
®
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_82901

inputs(
embedding_lookup_82895:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_82895Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/82895*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/82895*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶	
™
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_83849

inputs(
embedding_lookup_83843:
identityИҐembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€є
embedding_lookupResourceGatherembedding_lookup_83843Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/83843*+
_output_shapes
:€€€€€€€€€*
dtype0°
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/83843*+
_output_shapes
:€€€€€€€€€Б
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
Ч
'__inference_dense_1_layer_call_fn_83958

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83019p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
І	
Б
+__inference_concatenate_layer_call_fn_83919
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identityв
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_82989`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/4
Љ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_83899

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
М
i
<__inference_z_layer_call_and_return_conditional_losses_83147

inputs
inputs_1
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≥
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0*

seed**
seed2њ©∞Ц
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:€€€€€€€€€E
ExpExpmul:z:0*
T0*'
_output_shapes
:€€€€€€€€€Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѓ
В
.__inference_embedding_RACE_layer_call_fn_83822

inputs
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_82929s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
Ф
%__inference_dense_layer_call_fn_83938

inputs
unknown:	А
	unknown_0:	А
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83002p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
‘a
√

B__inference_encoder_layer_call_and_return_conditional_losses_83798
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous:
(embedding_marital_embedding_lookup_83719:<
*embedding_ethnicity_embedding_lookup_83725:7
%embedding_race_embedding_lookup_83731:9
'embedding_gender_embedding_lookup_83737:7
$dense_matmul_readvariableop_resource:	А4
%dense_biasadd_readvariableop_resource:	А:
&dense_1_matmul_readvariableop_resource:
АА6
'dense_1_biasadd_readvariableop_resource:	А8
%z_mean_matmul_readvariableop_resource:	А4
&z_mean_biasadd_readvariableop_resource:;
(z_log_var_matmul_readvariableop_resource:	А7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ$embedding_ETHNICITY/embedding_lookupҐ!embedding_GENDER/embedding_lookupҐ"embedding_MARITAL/embedding_lookupҐembedding_RACE/embedding_lookupҐ z_log_var/BiasAdd/ReadVariableOpҐz_log_var/MatMul/ReadVariableOpҐz_mean/BiasAdd/ReadVariableOpҐz_mean/MatMul/ReadVariableOpu
embedding_MARITAL/CastCastinputs_input_marital*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€Б
"embedding_MARITAL/embedding_lookupResourceGather(embedding_marital_embedding_lookup_83719embedding_MARITAL/Cast:y:0*
Tindices0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/83719*+
_output_shapes
:€€€€€€€€€*
dtype0„
+embedding_MARITAL/embedding_lookup/IdentityIdentity+embedding_MARITAL/embedding_lookup:output:0*
T0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/83719*+
_output_shapes
:€€€€€€€€€•
-embedding_MARITAL/embedding_lookup/Identity_1Identity4embedding_MARITAL/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€y
embedding_ETHNICITY/CastCastinputs_input_ethnicity*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€Й
$embedding_ETHNICITY/embedding_lookupResourceGather*embedding_ethnicity_embedding_lookup_83725embedding_ETHNICITY/Cast:y:0*
Tindices0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/83725*+
_output_shapes
:€€€€€€€€€*
dtype0Ё
-embedding_ETHNICITY/embedding_lookup/IdentityIdentity-embedding_ETHNICITY/embedding_lookup:output:0*
T0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/83725*+
_output_shapes
:€€€€€€€€€©
/embedding_ETHNICITY/embedding_lookup/Identity_1Identity6embedding_ETHNICITY/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€o
embedding_RACE/CastCastinputs_input_race*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€х
embedding_RACE/embedding_lookupResourceGather%embedding_race_embedding_lookup_83731embedding_RACE/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/83731*+
_output_shapes
:€€€€€€€€€*
dtype0ќ
(embedding_RACE/embedding_lookup/IdentityIdentity(embedding_RACE/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/83731*+
_output_shapes
:€€€€€€€€€Я
*embedding_RACE/embedding_lookup/Identity_1Identity1embedding_RACE/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€s
embedding_GENDER/CastCastinputs_input_gender*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€э
!embedding_GENDER/embedding_lookupResourceGather'embedding_gender_embedding_lookup_83737embedding_GENDER/Cast:y:0*
Tindices0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/83737*+
_output_shapes
:€€€€€€€€€*
dtype0‘
*embedding_GENDER/embedding_lookup/IdentityIdentity*embedding_GENDER/embedding_lookup:output:0*
T0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/83737*+
_output_shapes
:€€€€€€€€€£
,embedding_GENDER/embedding_lookup/Identity_1Identity3embedding_GENDER/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ы
flatten/ReshapeReshape5embedding_GENDER/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Э
flatten_1/ReshapeReshape3embedding_RACE/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ґ
flatten_2/ReshapeReshape8embedding_ETHNICITY/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   †
flatten_3/ReshapeReshape6embedding_MARITAL/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0inputs_input_continuous concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0М
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
z_mean/MatMulMatMuldense_1/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Й
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0С
z_log_var/MatMulMatMuldense_1/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€N
z/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:_
z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: a
z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
z/strided_sliceStridedSlicez/Shape:output:0z/strided_slice/stack:output:0 z/strided_slice/stack_1:output:0 z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
z/random_normal/shapePackz/strided_slice:output:0 z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:Y
z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    [
z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ґ
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0*

seed**
seed2†ЙMЬ
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
z/random_normalAddV2z/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€L
z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
z/mulMulz/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€I
z/ExpExp	z/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€`
z/mul_1Mul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€f
z/addAddV2z_mean/BiasAdd:output:0z/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	z/add:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Џ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^embedding_ETHNICITY/embedding_lookup"^embedding_GENDER/embedding_lookup#^embedding_MARITAL/embedding_lookup ^embedding_RACE/embedding_lookup!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2L
$embedding_ETHNICITY/embedding_lookup$embedding_ETHNICITY/embedding_lookup2F
!embedding_GENDER/embedding_lookup!embedding_GENDER/embedding_lookup2H
"embedding_MARITAL/embedding_lookup"embedding_MARITAL/embedding_lookup2B
embedding_RACE/embedding_lookupembedding_RACE/embedding_lookup2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:_ [
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs/input_continuous
‘a
√

B__inference_encoder_layer_call_and_return_conditional_losses_83711
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous:
(embedding_marital_embedding_lookup_83632:<
*embedding_ethnicity_embedding_lookup_83638:7
%embedding_race_embedding_lookup_83644:9
'embedding_gender_embedding_lookup_83650:7
$dense_matmul_readvariableop_resource:	А4
%dense_biasadd_readvariableop_resource:	А:
&dense_1_matmul_readvariableop_resource:
АА6
'dense_1_biasadd_readvariableop_resource:	А8
%z_mean_matmul_readvariableop_resource:	А4
&z_mean_biasadd_readvariableop_resource:;
(z_log_var_matmul_readvariableop_resource:	А7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐ$embedding_ETHNICITY/embedding_lookupҐ!embedding_GENDER/embedding_lookupҐ"embedding_MARITAL/embedding_lookupҐembedding_RACE/embedding_lookupҐ z_log_var/BiasAdd/ReadVariableOpҐz_log_var/MatMul/ReadVariableOpҐz_mean/BiasAdd/ReadVariableOpҐz_mean/MatMul/ReadVariableOpu
embedding_MARITAL/CastCastinputs_input_marital*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€Б
"embedding_MARITAL/embedding_lookupResourceGather(embedding_marital_embedding_lookup_83632embedding_MARITAL/Cast:y:0*
Tindices0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/83632*+
_output_shapes
:€€€€€€€€€*
dtype0„
+embedding_MARITAL/embedding_lookup/IdentityIdentity+embedding_MARITAL/embedding_lookup:output:0*
T0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/83632*+
_output_shapes
:€€€€€€€€€•
-embedding_MARITAL/embedding_lookup/Identity_1Identity4embedding_MARITAL/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€y
embedding_ETHNICITY/CastCastinputs_input_ethnicity*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€Й
$embedding_ETHNICITY/embedding_lookupResourceGather*embedding_ethnicity_embedding_lookup_83638embedding_ETHNICITY/Cast:y:0*
Tindices0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/83638*+
_output_shapes
:€€€€€€€€€*
dtype0Ё
-embedding_ETHNICITY/embedding_lookup/IdentityIdentity-embedding_ETHNICITY/embedding_lookup:output:0*
T0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/83638*+
_output_shapes
:€€€€€€€€€©
/embedding_ETHNICITY/embedding_lookup/Identity_1Identity6embedding_ETHNICITY/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€o
embedding_RACE/CastCastinputs_input_race*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€х
embedding_RACE/embedding_lookupResourceGather%embedding_race_embedding_lookup_83644embedding_RACE/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/83644*+
_output_shapes
:€€€€€€€€€*
dtype0ќ
(embedding_RACE/embedding_lookup/IdentityIdentity(embedding_RACE/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/83644*+
_output_shapes
:€€€€€€€€€Я
*embedding_RACE/embedding_lookup/Identity_1Identity1embedding_RACE/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€s
embedding_GENDER/CastCastinputs_input_gender*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€э
!embedding_GENDER/embedding_lookupResourceGather'embedding_gender_embedding_lookup_83650embedding_GENDER/Cast:y:0*
Tindices0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/83650*+
_output_shapes
:€€€€€€€€€*
dtype0‘
*embedding_GENDER/embedding_lookup/IdentityIdentity*embedding_GENDER/embedding_lookup:output:0*
T0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/83650*+
_output_shapes
:€€€€€€€€€£
,embedding_GENDER/embedding_lookup/Identity_1Identity3embedding_GENDER/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ы
flatten/ReshapeReshape5embedding_GENDER/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Э
flatten_1/ReshapeReshape3embedding_RACE/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ґ
flatten_2/ReshapeReshape8embedding_ETHNICITY/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   †
flatten_3/ReshapeReshape6embedding_MARITAL/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0inputs_input_continuous concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0М
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
z_mean/MatMulMatMuldense_1/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€А
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Й
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0С
z_log_var/MatMulMatMuldense_1/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€N
z/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:_
z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: a
z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
z/strided_sliceStridedSlicez/Shape:output:0z/strided_slice/stack:output:0 z/strided_slice/stack_1:output:0 z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
z/random_normal/shapePackz/strided_slice:output:0 z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:Y
z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    [
z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ґ
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0*

seed**
seed2їІЬ
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
z/random_normalAddV2z/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€L
z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
z/mulMulz/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€I
z/ExpExp	z/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€`
z/mul_1Mul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€f
z/addAddV2z_mean/BiasAdd:output:0z/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	z/add:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Џ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^embedding_ETHNICITY/embedding_lookup"^embedding_GENDER/embedding_lookup#^embedding_MARITAL/embedding_lookup ^embedding_RACE/embedding_lookup!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2L
$embedding_ETHNICITY/embedding_lookup$embedding_ETHNICITY/embedding_lookup2F
!embedding_GENDER/embedding_lookup!embedding_GENDER/embedding_lookup2H
"embedding_MARITAL/embedding_lookup"embedding_MARITAL/embedding_lookup2B
embedding_RACE/embedding_lookupembedding_RACE/embedding_lookup2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:_ [
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs/input_continuous
Я

у
@__inference_dense_layer_call_and_return_conditional_losses_83949

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
з
j
!__inference_z_layer_call_fn_84019
inputs_0
inputs_1
identityИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_83147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
—B
±
B__inference_encoder_layer_call_and_return_conditional_losses_83084

inputs
inputs_1
inputs_2
inputs_3
inputs_4)
embedding_marital_82902:+
embedding_ethnicity_82916:&
embedding_race_82930:(
embedding_gender_82944:
dense_83003:	А
dense_83005:	А!
dense_1_83020:
АА
dense_1_83022:	А
z_mean_83036:	А
z_mean_83038:"
z_log_var_83052:	А
z_log_var_83054:
identity

identity_1

identity_2ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ+embedding_ETHNICITY/StatefulPartitionedCallҐ(embedding_GENDER/StatefulPartitionedCallҐ)embedding_MARITAL/StatefulPartitionedCallҐ&embedding_RACE/StatefulPartitionedCallҐz/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCall€
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_marital_82902*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_82901Г
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_ethnicity_82916*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_82915ц
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_race_82930*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_82929ь
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_gender_82944*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_82943г
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_82953е
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_82961к
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_82969и
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_82977‘
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_82989Г
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_83003dense_83005*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83002Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_83020dense_1_83022*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83019К
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_83036z_mean_83038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_83035Ц
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_83052z_log_var_83054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_83051К
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_83079v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ч
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+embedding_ETHNICITY/StatefulPartitionedCall+embedding_ETHNICITY/StatefulPartitionedCall2T
(embedding_GENDER/StatefulPartitionedCall(embedding_GENDER/StatefulPartitionedCall2V
)embedding_MARITAL/StatefulPartitionedCall)embedding_MARITAL/StatefulPartitionedCall2P
&embedding_RACE/StatefulPartitionedCall&embedding_RACE/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Љm
Њ
 __inference__wrapped_model_82876
input_ethnicity
input_gender
input_marital

input_race
input_continuousB
0encoder_embedding_marital_embedding_lookup_82797:D
2encoder_embedding_ethnicity_embedding_lookup_82803:?
-encoder_embedding_race_embedding_lookup_82809:A
/encoder_embedding_gender_embedding_lookup_82815:?
,encoder_dense_matmul_readvariableop_resource:	А<
-encoder_dense_biasadd_readvariableop_resource:	АB
.encoder_dense_1_matmul_readvariableop_resource:
АА>
/encoder_dense_1_biasadd_readvariableop_resource:	А@
-encoder_z_mean_matmul_readvariableop_resource:	А<
.encoder_z_mean_biasadd_readvariableop_resource:C
0encoder_z_log_var_matmul_readvariableop_resource:	А?
1encoder_z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐ$encoder/dense/BiasAdd/ReadVariableOpҐ#encoder/dense/MatMul/ReadVariableOpҐ&encoder/dense_1/BiasAdd/ReadVariableOpҐ%encoder/dense_1/MatMul/ReadVariableOpҐ,encoder/embedding_ETHNICITY/embedding_lookupҐ)encoder/embedding_GENDER/embedding_lookupҐ*encoder/embedding_MARITAL/embedding_lookupҐ'encoder/embedding_RACE/embedding_lookupҐ(encoder/z_log_var/BiasAdd/ReadVariableOpҐ'encoder/z_log_var/MatMul/ReadVariableOpҐ%encoder/z_mean/BiasAdd/ReadVariableOpҐ$encoder/z_mean/MatMul/ReadVariableOpv
encoder/embedding_MARITAL/CastCastinput_marital*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€°
*encoder/embedding_MARITAL/embedding_lookupResourceGather0encoder_embedding_marital_embedding_lookup_82797"encoder/embedding_MARITAL/Cast:y:0*
Tindices0*C
_class9
75loc:@encoder/embedding_MARITAL/embedding_lookup/82797*+
_output_shapes
:€€€€€€€€€*
dtype0п
3encoder/embedding_MARITAL/embedding_lookup/IdentityIdentity3encoder/embedding_MARITAL/embedding_lookup:output:0*
T0*C
_class9
75loc:@encoder/embedding_MARITAL/embedding_lookup/82797*+
_output_shapes
:€€€€€€€€€µ
5encoder/embedding_MARITAL/embedding_lookup/Identity_1Identity<encoder/embedding_MARITAL/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€z
 encoder/embedding_ETHNICITY/CastCastinput_ethnicity*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€©
,encoder/embedding_ETHNICITY/embedding_lookupResourceGather2encoder_embedding_ethnicity_embedding_lookup_82803$encoder/embedding_ETHNICITY/Cast:y:0*
Tindices0*E
_class;
97loc:@encoder/embedding_ETHNICITY/embedding_lookup/82803*+
_output_shapes
:€€€€€€€€€*
dtype0х
5encoder/embedding_ETHNICITY/embedding_lookup/IdentityIdentity5encoder/embedding_ETHNICITY/embedding_lookup:output:0*
T0*E
_class;
97loc:@encoder/embedding_ETHNICITY/embedding_lookup/82803*+
_output_shapes
:€€€€€€€€€є
7encoder/embedding_ETHNICITY/embedding_lookup/Identity_1Identity>encoder/embedding_ETHNICITY/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€p
encoder/embedding_RACE/CastCast
input_race*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€Х
'encoder/embedding_RACE/embedding_lookupResourceGather-encoder_embedding_race_embedding_lookup_82809encoder/embedding_RACE/Cast:y:0*
Tindices0*@
_class6
42loc:@encoder/embedding_RACE/embedding_lookup/82809*+
_output_shapes
:€€€€€€€€€*
dtype0ж
0encoder/embedding_RACE/embedding_lookup/IdentityIdentity0encoder/embedding_RACE/embedding_lookup:output:0*
T0*@
_class6
42loc:@encoder/embedding_RACE/embedding_lookup/82809*+
_output_shapes
:€€€€€€€€€ѓ
2encoder/embedding_RACE/embedding_lookup/Identity_1Identity9encoder/embedding_RACE/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€t
encoder/embedding_GENDER/CastCastinput_gender*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€Э
)encoder/embedding_GENDER/embedding_lookupResourceGather/encoder_embedding_gender_embedding_lookup_82815!encoder/embedding_GENDER/Cast:y:0*
Tindices0*B
_class8
64loc:@encoder/embedding_GENDER/embedding_lookup/82815*+
_output_shapes
:€€€€€€€€€*
dtype0м
2encoder/embedding_GENDER/embedding_lookup/IdentityIdentity2encoder/embedding_GENDER/embedding_lookup:output:0*
T0*B
_class8
64loc:@encoder/embedding_GENDER/embedding_lookup/82815*+
_output_shapes
:€€€€€€€€€≥
4encoder/embedding_GENDER/embedding_lookup/Identity_1Identity;encoder/embedding_GENDER/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€f
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ≥
encoder/flatten/ReshapeReshape=encoder/embedding_GENDER/embedding_lookup/Identity_1:output:0encoder/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   µ
encoder/flatten_1/ReshapeReshape;encoder/embedding_RACE/embedding_lookup/Identity_1:output:0 encoder/flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ї
encoder/flatten_2/ReshapeReshape@encoder/embedding_ETHNICITY/embedding_lookup/Identity_1:output:0 encoder/flatten_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
encoder/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Є
encoder/flatten_3/ReshapeReshape>encoder/embedding_MARITAL/embedding_lookup/Identity_1:output:0 encoder/flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
encoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ђ
encoder/concatenate/concatConcatV2 encoder/flatten/Reshape:output:0"encoder/flatten_1/Reshape:output:0"encoder/flatten_2/Reshape:output:0"encoder/flatten_3/Reshape:output:0input_continuous(encoder/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€С
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0£
encoder/dense/MatMulMatMul#encoder/concatenate/concat:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АП
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0°
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аm
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЦ
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0§
encoder/dense_1/MatMulMatMul encoder/dense/Relu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АУ
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0І
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аq
encoder/dense_1/ReluRelu encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АУ
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0£
encoder/z_mean/MatMulMatMul"encoder/dense_1/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Р
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Щ
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0©
encoder/z_log_var/MatMulMatMul"encoder/dense_1/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
encoder/z/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:g
encoder/z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
encoder/z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
encoder/z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
encoder/z/strided_sliceStridedSliceencoder/z/Shape:output:0&encoder/z/strided_slice/stack:output:0(encoder/z/strided_slice/stack_1:output:0(encoder/z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
encoder/z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Я
encoder/z/random_normal/shapePack encoder/z/strided_slice:output:0(encoder/z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:a
encoder/z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    c
encoder/z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?«
,encoder/z/random_normal/RandomStandardNormalRandomStandardNormal&encoder/z/random_normal/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0*

seed**
seed2•ґпі
encoder/z/random_normal/mulMul5encoder/z/random_normal/RandomStandardNormal:output:0'encoder/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
encoder/z/random_normalAddV2encoder/z/random_normal/mul:z:0%encoder/z/random_normal/mean:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
encoder/z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Д
encoder/z/mulMulencoder/z/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Y
encoder/z/ExpExpencoder/z/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€x
encoder/z/mul_1Mulencoder/z/Exp:y:0encoder/z/random_normal:z:0*
T0*'
_output_shapes
:€€€€€€€€€~
encoder/z/addAddV2encoder/z_mean/BiasAdd:output:0encoder/z/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentityencoder/z/add:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€s

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€p

Identity_2Identityencoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ї
NoOpNoOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp-^encoder/embedding_ETHNICITY/embedding_lookup*^encoder/embedding_GENDER/embedding_lookup+^encoder/embedding_MARITAL/embedding_lookup(^encoder/embedding_RACE/embedding_lookup)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp2\
,encoder/embedding_ETHNICITY/embedding_lookup,encoder/embedding_ETHNICITY/embedding_lookup2V
)encoder/embedding_GENDER/embedding_lookup)encoder/embedding_GENDER/embedding_lookup2X
*encoder/embedding_MARITAL/embedding_lookup*encoder/embedding_MARITAL/embedding_lookup2R
'encoder/embedding_RACE/embedding_lookup'encoder/embedding_RACE/embedding_lookup2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp2N
%encoder/z_mean/BiasAdd/ReadVariableOp%encoder/z_mean/BiasAdd/ReadVariableOp2L
$encoder/z_mean/MatMul/ReadVariableOp$encoder/z_mean/MatMul/ReadVariableOp:X T
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinput_continuous
Э$
Ћ
__inference__traced_save_84128
file_prefix:
6savev2_embedding_gender_embeddings_read_readvariableop8
4savev2_embedding_race_embeddings_read_readvariableop=
9savev2_embedding_ethnicity_embeddings_read_readvariableop;
7savev2_embedding_marital_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ≤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ
value—BќB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЗ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B м
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_embedding_gender_embeddings_read_readvariableop4savev2_embedding_race_embeddings_read_readvariableop9savev2_embedding_ethnicity_embeddings_read_readvariableop7savev2_embedding_marital_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ж
_input_shapesu
s: :::::	А:А:
АА:А:	А::	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%	!

_output_shapes
:	А: 


_output_shapes
::%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: 
—B
±
B__inference_encoder_layer_call_and_return_conditional_losses_83347

inputs
inputs_1
inputs_2
inputs_3
inputs_4)
embedding_marital_83306:+
embedding_ethnicity_83309:&
embedding_race_83312:(
embedding_gender_83315:
dense_83323:	А
dense_83325:	А!
dense_1_83328:
АА
dense_1_83330:	А
z_mean_83333:	А
z_mean_83335:"
z_log_var_83338:	А
z_log_var_83340:
identity

identity_1

identity_2ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ+embedding_ETHNICITY/StatefulPartitionedCallҐ(embedding_GENDER/StatefulPartitionedCallҐ)embedding_MARITAL/StatefulPartitionedCallҐ&embedding_RACE/StatefulPartitionedCallҐz/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCall€
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_marital_83306*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_82901Г
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_ethnicity_83309*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_82915ц
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_race_83312*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_82929ь
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_gender_83315*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_82943г
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_82953е
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_82961к
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_82969и
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_82977‘
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_82989Г
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_83323dense_83325*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83002Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_83328dense_1_83330*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83019К
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_83333z_mean_83335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_83035Ц
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_83338z_log_var_83340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_83051К
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_83147v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ч
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+embedding_ETHNICITY/StatefulPartitionedCall+embedding_ETHNICITY/StatefulPartitionedCall2T
(embedding_GENDER/StatefulPartitionedCall(embedding_GENDER/StatefulPartitionedCall2V
)embedding_MARITAL/StatefulPartitionedCall)embedding_MARITAL/StatefulPartitionedCall2P
&embedding_RACE/StatefulPartitionedCall&embedding_RACE/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
§
C
'__inference_flatten_layer_call_fn_83871

inputs
identity∞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_82953`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠C
Ќ
B__inference_encoder_layer_call_and_return_conditional_losses_83511
input_ethnicity
input_gender
input_marital

input_race
input_continuous)
embedding_marital_83470:+
embedding_ethnicity_83473:&
embedding_race_83476:(
embedding_gender_83479:
dense_83487:	А
dense_83489:	А!
dense_1_83492:
АА
dense_1_83494:	А
z_mean_83497:	А
z_mean_83499:"
z_log_var_83502:	А
z_log_var_83504:
identity

identity_1

identity_2ИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐ+embedding_ETHNICITY/StatefulPartitionedCallҐ(embedding_GENDER/StatefulPartitionedCallҐ)embedding_MARITAL/StatefulPartitionedCallҐ&embedding_RACE/StatefulPartitionedCallҐz/StatefulPartitionedCallҐ!z_log_var/StatefulPartitionedCallҐz_mean/StatefulPartitionedCallД
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinput_maritalembedding_marital_83470*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_82901М
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityembedding_ethnicity_83473*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_82915ш
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCall
input_raceembedding_race_83476*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_82929А
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinput_genderembedding_gender_83479*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_82943г
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_82953е
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_82961к
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_82969и
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_82977№
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0input_continuous*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_82989Г
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_83487dense_83489*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_83002Н
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_83492dense_1_83494*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_83019К
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_83497z_mean_83499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_83035Ц
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_83502z_log_var_83504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_83051К
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_83147v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ч
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+embedding_ETHNICITY/StatefulPartitionedCall+embedding_ETHNICITY/StatefulPartitionedCall2T
(embedding_GENDER/StatefulPartitionedCall(embedding_GENDER/StatefulPartitionedCall2V
)embedding_MARITAL/StatefulPartitionedCall)embedding_MARITAL/StatefulPartitionedCall2P
&embedding_RACE/StatefulPartitionedCall&embedding_RACE/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:X T
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinput_continuous
»
Ч
)__inference_z_log_var_layer_call_fn_83997

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_83051o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ћ	
ц
D__inference_z_log_var_layer_call_and_return_conditional_losses_84007

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ї
^
B__inference_flatten_layer_call_and_return_conditional_losses_82953

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
з
j
!__inference_z_layer_call_fn_84013
inputs_0
inputs_1
identityИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_83079o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*—
serving_defaultљ
K
input_ETHNICITY8
!serving_default_input_ETHNICITY:0€€€€€€€€€
E
input_GENDER5
serving_default_input_GENDER:0€€€€€€€€€
G
input_MARITAL6
serving_default_input_MARITAL:0€€€€€€€€€
A

input_RACE3
serving_default_input_RACE:0€€€€€€€€€
M
input_continuous9
"serving_default_input_continuous:0€€€€€€€€€5
z0
StatefulPartitionedCall:0€€€€€€€€€=
	z_log_var0
StatefulPartitionedCall:1€€€€€€€€€:
z_mean0
StatefulPartitionedCall:2€€€€€€€€€tensorflow/serving/predict:а 
§
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer_with_weights-5
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
µ
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"
embeddings"
_tf_keras_layer
µ
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
embeddings"
_tf_keras_layer
µ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0
embeddings"
_tf_keras_layer
µ
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7
embeddings"
_tf_keras_layer
•
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
•
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
•
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
•
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
•
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
ї
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias"
_tf_keras_layer
ї
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias"
_tf_keras_layer
ї
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias"
_tf_keras_layer
•
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
v
"0
)1
02
73
\4
]5
d6
e7
l8
m9
t10
u11"
trackable_list_wrapper
v
"0
)1
02
73
\4
]5
d6
e7
l8
m9
t10
u11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ћ
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ў
Бtrace_0
Вtrace_1
Гtrace_2
Дtrace_32ж
'__inference_encoder_layer_call_fn_83115
'__inference_encoder_layer_call_fn_83587
'__inference_encoder_layer_call_fn_83624
'__inference_encoder_layer_call_fn_83415њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zБtrace_0zВtrace_1zГtrace_2zДtrace_3
≈
Еtrace_0
Жtrace_1
Зtrace_2
Иtrace_32“
B__inference_encoder_layer_call_and_return_conditional_losses_83711
B__inference_encoder_layer_call_and_return_conditional_losses_83798
B__inference_encoder_layer_call_and_return_conditional_losses_83463
B__inference_encoder_layer_call_and_return_conditional_losses_83511њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЕtrace_0zЖtrace_1zЗtrace_2zИtrace_3
ОBЛ
 __inference__wrapped_model_82876input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
-
Йserving_default"
signature_map
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ц
Пtrace_02„
0__inference_embedding_GENDER_layer_call_fn_83805Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zПtrace_0
С
Рtrace_02т
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_83815Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zРtrace_0
-:+2embedding_GENDER/embeddings
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ф
Цtrace_02’
.__inference_embedding_RACE_layer_call_fn_83822Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЦtrace_0
П
Чtrace_02р
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_83832Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЧtrace_0
+:)2embedding_RACE/embeddings
'
00"
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
щ
Эtrace_02Џ
3__inference_embedding_ETHNICITY_layer_call_fn_83839Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЭtrace_0
Ф
Юtrace_02х
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_83849Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЮtrace_0
0:.2embedding_ETHNICITY/embeddings
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
ч
§trace_02Ў
1__inference_embedding_MARITAL_layer_call_fn_83856Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0
Т
•trace_02у
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_83866Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z•trace_0
.:,2embedding_MARITAL/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
н
Ђtrace_02ќ
'__inference_flatten_layer_call_fn_83871Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0
И
ђtrace_02й
B__inference_flatten_layer_call_and_return_conditional_losses_83877Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zђtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
п
≤trace_02–
)__inference_flatten_1_layer_call_fn_83882Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≤trace_0
К
≥trace_02л
D__inference_flatten_1_layer_call_and_return_conditional_losses_83888Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≥trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
п
єtrace_02–
)__inference_flatten_2_layer_call_fn_83893Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zєtrace_0
К
Їtrace_02л
D__inference_flatten_2_layer_call_and_return_conditional_losses_83899Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЇtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
п
јtrace_02–
)__inference_flatten_3_layer_call_fn_83904Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zјtrace_0
К
Ѕtrace_02л
D__inference_flatten_3_layer_call_and_return_conditional_losses_83910Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
с
«trace_02“
+__inference_concatenate_layer_call_fn_83919Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0
М
»trace_02н
F__inference_concatenate_layer_call_and_return_conditional_losses_83929Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z»trace_0
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
л
ќtrace_02ћ
%__inference_dense_layer_call_fn_83938Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zќtrace_0
Ж
ѕtrace_02з
@__inference_dense_layer_call_and_return_conditional_losses_83949Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѕtrace_0
:	А2dense/kernel
:А2
dense/bias
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
н
’trace_02ќ
'__inference_dense_1_layer_call_fn_83958Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z’trace_0
И
÷trace_02й
B__inference_dense_1_layer_call_and_return_conditional_losses_83969Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0
": 
АА2dense_1/kernel
:А2dense_1/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
м
№trace_02Ќ
&__inference_z_mean_layer_call_fn_83978Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z№trace_0
З
Ёtrace_02и
A__inference_z_mean_layer_call_and_return_conditional_losses_83988Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЁtrace_0
 :	А2z_mean/kernel
:2z_mean/bias
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ёnon_trainable_variables
яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
п
гtrace_02–
)__inference_z_log_var_layer_call_fn_83997Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zгtrace_0
К
дtrace_02л
D__inference_z_log_var_layer_call_and_return_conditional_losses_84007Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zдtrace_0
#:!	А2z_log_var/kernel
:2z_log_var/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
√
кtrace_0
лtrace_12И
!__inference_z_layer_call_fn_84013
!__inference_z_layer_call_fn_84019њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zкtrace_0zлtrace_1
щ
мtrace_0
нtrace_12Њ
<__inference_z_layer_call_and_return_conditional_losses_84041
<__inference_z_layer_call_and_return_conditional_losses_84063њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zмtrace_0zнtrace_1
 "
trackable_list_wrapper
Ѓ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЉBє
'__inference_encoder_layer_call_fn_83115input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
яB№
'__inference_encoder_layer_call_fn_83587inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
яB№
'__inference_encoder_layer_call_fn_83624inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЉBє
'__inference_encoder_layer_call_fn_83415input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
B__inference_encoder_layer_call_and_return_conditional_losses_83711inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
B__inference_encoder_layer_call_and_return_conditional_losses_83798inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„B‘
B__inference_encoder_layer_call_and_return_conditional_losses_83463input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„B‘
B__inference_encoder_layer_call_and_return_conditional_losses_83511input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЛBИ
#__inference_signature_wrapper_83550input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
0__inference_embedding_GENDER_layer_call_fn_83805inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_83815inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBя
.__inference_embedding_RACE_layer_call_fn_83822inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_83832inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_embedding_ETHNICITY_layer_call_fn_83839inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_83849inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
1__inference_embedding_MARITAL_layer_call_fn_83856inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_83866inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
џBЎ
'__inference_flatten_layer_call_fn_83871inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
B__inference_flatten_layer_call_and_return_conditional_losses_83877inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_flatten_1_layer_call_fn_83882inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_flatten_1_layer_call_and_return_conditional_losses_83888inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_flatten_2_layer_call_fn_83893inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_flatten_2_layer_call_and_return_conditional_losses_83899inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_flatten_3_layer_call_fn_83904inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_flatten_3_layer_call_and_return_conditional_losses_83910inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЙBЖ
+__inference_concatenate_layer_call_fn_83919inputs/0inputs/1inputs/2inputs/3inputs/4"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
§B°
F__inference_concatenate_layer_call_and_return_conditional_losses_83929inputs/0inputs/1inputs/2inputs/3inputs/4"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ўB÷
%__inference_dense_layer_call_fn_83938inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
@__inference_dense_layer_call_and_return_conditional_losses_83949inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
џBЎ
'__inference_dense_1_layer_call_fn_83958inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
B__inference_dense_1_layer_call_and_return_conditional_losses_83969inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
&__inference_z_mean_layer_call_fn_83978inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
A__inference_z_mean_layer_call_and_return_conditional_losses_83988inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_z_log_var_layer_call_fn_83997inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_z_log_var_layer_call_and_return_conditional_losses_84007inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
!__inference_z_layer_call_fn_84013inputs/0inputs/1"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
!__inference_z_layer_call_fn_84019inputs/0inputs/1"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
<__inference_z_layer_call_and_return_conditional_losses_84041inputs/0inputs/1"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
<__inference_z_layer_call_and_return_conditional_losses_84063inputs/0inputs/1"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ш
 __inference__wrapped_model_82876”70)"\]delmtuїҐЈ
ѓҐЂ
®™§
<
input_ETHNICITY)К&
input_ETHNICITY€€€€€€€€€
6
input_GENDER&К#
input_GENDER€€€€€€€€€
8
input_MARITAL'К$
input_MARITAL€€€€€€€€€
2

input_RACE$К!

input_RACE€€€€€€€€€
>
input_continuous*К'
input_continuous€€€€€€€€€
™ "Д™А
 
zК
z€€€€€€€€€
0
	z_log_var#К 
	z_log_var€€€€€€€€€
*
z_mean К
z_mean€€€€€€€€€ј
F__inference_concatenate_layer_call_and_return_conditional_losses_83929хЋҐ«
њҐї
ЄЪі
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ш
+__inference_concatenate_layer_call_fn_83919иЋҐ«
њҐї
ЄЪі
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€
"К
inputs/4€€€€€€€€€
™ "К€€€€€€€€€§
B__inference_dense_1_layer_call_and_return_conditional_losses_83969^de0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ |
'__inference_dense_1_layer_call_fn_83958Qde0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€А°
@__inference_dense_layer_call_and_return_conditional_losses_83949]\]/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€А
Ъ y
%__inference_dense_layer_call_fn_83938P\]/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€А±
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_83849_0/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ Й
3__inference_embedding_ETHNICITY_layer_call_fn_83839R0/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Ѓ
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_83815_"/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ж
0__inference_embedding_GENDER_layer_call_fn_83805R"/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ѓ
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_83866_7/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ З
1__inference_embedding_MARITAL_layer_call_fn_83856R7/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ђ
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_83832_)/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ Д
.__inference_embedding_RACE_layer_call_fn_83822R)/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€З
B__inference_encoder_layer_call_and_return_conditional_losses_83463ј70)"\]delmtu√Ґњ
ЈҐ≥
®™§
<
input_ETHNICITY)К&
input_ETHNICITY€€€€€€€€€
6
input_GENDER&К#
input_GENDER€€€€€€€€€
8
input_MARITAL'К$
input_MARITAL€€€€€€€€€
2

input_RACE$К!

input_RACE€€€€€€€€€
>
input_continuous*К'
input_continuous€€€€€€€€€
p 

 
™ "jҐg
`Ъ]
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
К
0/2€€€€€€€€€
Ъ З
B__inference_encoder_layer_call_and_return_conditional_losses_83511ј70)"\]delmtu√Ґњ
ЈҐ≥
®™§
<
input_ETHNICITY)К&
input_ETHNICITY€€€€€€€€€
6
input_GENDER&К#
input_GENDER€€€€€€€€€
8
input_MARITAL'К$
input_MARITAL€€€€€€€€€
2

input_RACE$К!

input_RACE€€€€€€€€€
>
input_continuous*К'
input_continuous€€€€€€€€€
p

 
™ "jҐg
`Ъ]
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
К
0/2€€€€€€€€€
Ъ ™
B__inference_encoder_layer_call_and_return_conditional_losses_83711г70)"\]delmtuжҐв
ЏҐ÷
Ћ™«
C
input_ETHNICITY0К-
inputs/input_ETHNICITY€€€€€€€€€
=
input_GENDER-К*
inputs/input_GENDER€€€€€€€€€
?
input_MARITAL.К+
inputs/input_MARITAL€€€€€€€€€
9

input_RACE+К(
inputs/input_RACE€€€€€€€€€
E
input_continuous1К.
inputs/input_continuous€€€€€€€€€
p 

 
™ "jҐg
`Ъ]
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
К
0/2€€€€€€€€€
Ъ ™
B__inference_encoder_layer_call_and_return_conditional_losses_83798г70)"\]delmtuжҐв
ЏҐ÷
Ћ™«
C
input_ETHNICITY0К-
inputs/input_ETHNICITY€€€€€€€€€
=
input_GENDER-К*
inputs/input_GENDER€€€€€€€€€
?
input_MARITAL.К+
inputs/input_MARITAL€€€€€€€€€
9

input_RACE+К(
inputs/input_RACE€€€€€€€€€
E
input_continuous1К.
inputs/input_continuous€€€€€€€€€
p

 
™ "jҐg
`Ъ]
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
К
0/2€€€€€€€€€
Ъ №
'__inference_encoder_layer_call_fn_83115∞70)"\]delmtu√Ґњ
ЈҐ≥
®™§
<
input_ETHNICITY)К&
input_ETHNICITY€€€€€€€€€
6
input_GENDER&К#
input_GENDER€€€€€€€€€
8
input_MARITAL'К$
input_MARITAL€€€€€€€€€
2

input_RACE$К!

input_RACE€€€€€€€€€
>
input_continuous*К'
input_continuous€€€€€€€€€
p 

 
™ "ZЪW
К
0€€€€€€€€€
К
1€€€€€€€€€
К
2€€€€€€€€€№
'__inference_encoder_layer_call_fn_83415∞70)"\]delmtu√Ґњ
ЈҐ≥
®™§
<
input_ETHNICITY)К&
input_ETHNICITY€€€€€€€€€
6
input_GENDER&К#
input_GENDER€€€€€€€€€
8
input_MARITAL'К$
input_MARITAL€€€€€€€€€
2

input_RACE$К!

input_RACE€€€€€€€€€
>
input_continuous*К'
input_continuous€€€€€€€€€
p

 
™ "ZЪW
К
0€€€€€€€€€
К
1€€€€€€€€€
К
2€€€€€€€€€€
'__inference_encoder_layer_call_fn_83587”70)"\]delmtuжҐв
ЏҐ÷
Ћ™«
C
input_ETHNICITY0К-
inputs/input_ETHNICITY€€€€€€€€€
=
input_GENDER-К*
inputs/input_GENDER€€€€€€€€€
?
input_MARITAL.К+
inputs/input_MARITAL€€€€€€€€€
9

input_RACE+К(
inputs/input_RACE€€€€€€€€€
E
input_continuous1К.
inputs/input_continuous€€€€€€€€€
p 

 
™ "ZЪW
К
0€€€€€€€€€
К
1€€€€€€€€€
К
2€€€€€€€€€€
'__inference_encoder_layer_call_fn_83624”70)"\]delmtuжҐв
ЏҐ÷
Ћ™«
C
input_ETHNICITY0К-
inputs/input_ETHNICITY€€€€€€€€€
=
input_GENDER-К*
inputs/input_GENDER€€€€€€€€€
?
input_MARITAL.К+
inputs/input_MARITAL€€€€€€€€€
9

input_RACE+К(
inputs/input_RACE€€€€€€€€€
E
input_continuous1К.
inputs/input_continuous€€€€€€€€€
p

 
™ "ZЪW
К
0€€€€€€€€€
К
1€€€€€€€€€
К
2€€€€€€€€€§
D__inference_flatten_1_layer_call_and_return_conditional_losses_83888\3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_flatten_1_layer_call_fn_83882O3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "К€€€€€€€€€§
D__inference_flatten_2_layer_call_and_return_conditional_losses_83899\3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_flatten_2_layer_call_fn_83893O3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "К€€€€€€€€€§
D__inference_flatten_3_layer_call_and_return_conditional_losses_83910\3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_flatten_3_layer_call_fn_83904O3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "К€€€€€€€€€Ґ
B__inference_flatten_layer_call_and_return_conditional_losses_83877\3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
'__inference_flatten_layer_call_fn_83871O3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "К€€€€€€€€€ф
#__inference_signature_wrapper_83550ћ70)"\]delmtuіҐ∞
Ґ 
®™§
<
input_ETHNICITY)К&
input_ETHNICITY€€€€€€€€€
6
input_GENDER&К#
input_GENDER€€€€€€€€€
8
input_MARITAL'К$
input_MARITAL€€€€€€€€€
2

input_RACE$К!

input_RACE€€€€€€€€€
>
input_continuous*К'
input_continuous€€€€€€€€€"Д™А
 
zК
z€€€€€€€€€
0
	z_log_var#К 
	z_log_var€€€€€€€€€
*
z_mean К
z_mean€€€€€€€€€ћ
<__inference_z_layer_call_and_return_conditional_losses_84041ЛbҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€

 
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ћ
<__inference_z_layer_call_and_return_conditional_losses_84063ЛbҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€

 
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ £
!__inference_z_layer_call_fn_84013~bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€

 
p 
™ "К€€€€€€€€€£
!__inference_z_layer_call_fn_84019~bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€

 
p
™ "К€€€€€€€€€•
D__inference_z_log_var_layer_call_and_return_conditional_losses_84007]tu0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
)__inference_z_log_var_layer_call_fn_83997Ptu0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ґ
A__inference_z_mean_layer_call_and_return_conditional_losses_83988]lm0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
&__inference_z_mean_layer_call_fn_83978Plm0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€