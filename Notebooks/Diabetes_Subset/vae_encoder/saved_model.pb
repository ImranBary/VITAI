��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��	
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
shape:	�*!
shared_namez_log_var/kernel
v
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes
:	�*
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
shape:	�*
shared_namez_mean/kernel
p
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes
:	�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
�
embedding_MARITAL/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameembedding_MARITAL/embeddings
�
0embedding_MARITAL/embeddings/Read/ReadVariableOpReadVariableOpembedding_MARITAL/embeddings*
_output_shapes

:*
dtype0
�
embedding_ETHNICITY/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name embedding_ETHNICITY/embeddings
�
2embedding_ETHNICITY/embeddings/Read/ReadVariableOpReadVariableOpembedding_ETHNICITY/embeddings*
_output_shapes

:*
dtype0
�
embedding_RACE/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameembedding_RACE/embeddings
�
-embedding_RACE/embeddings/Read/ReadVariableOpReadVariableOpembedding_RACE/embeddings*
_output_shapes

:*
dtype0
�
embedding_GENDER/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameembedding_GENDER/embeddings
�
/embedding_GENDER/embeddings/Read/ReadVariableOpReadVariableOpembedding_GENDER/embeddings*
_output_shapes

:*
dtype0
�
serving_default_input_ETHNICITYPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������

serving_default_input_GENDERPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_input_MARITALPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
}
serving_default_input_RACEPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
 serving_default_input_continuousPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_ETHNICITYserving_default_input_GENDERserving_default_input_MARITALserving_default_input_RACE serving_default_input_continuousembedding_MARITAL/embeddingsembedding_ETHNICITY/embeddingsembedding_RACE/embeddingsembedding_GENDER/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_62860

NoOpNoOp
�P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�O
value�OB�O B�O
�
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
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"
embeddings*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
embeddings*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0
embeddings*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7
embeddings*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
* 
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias*
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias*
�
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
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 

"0*

"0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
oi
VARIABLE_VALUEembedding_GENDER/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

)0*

)0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
mg
VARIABLE_VALUEembedding_RACE/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

00*

00*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
rl
VARIABLE_VALUEembedding_ETHNICITY/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

70*

70*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
pj
VARIABLE_VALUEembedding_MARITAL/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

\0
]1*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEz_log_var/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
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
�
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
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_63438
�
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
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_63484��
�	
�
A__inference_z_mean_layer_call_and_return_conditional_losses_63298

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
<__inference_z_layer_call_and_return_conditional_losses_62457

inputs
inputs_1
identity�;
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
valueB:�
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
value	B :�
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
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2����
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_62263

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
E
)__inference_flatten_2_layer_call_fn_63203

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_62279`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�C
�
B__inference_encoder_layer_call_and_return_conditional_losses_62773
input_ethnicity
input_gender
input_marital

input_race
input_continuous)
embedding_marital_62732:+
embedding_ethnicity_62735:&
embedding_race_62738:(
embedding_gender_62741:
dense_62749:	�
dense_62751:	�!
dense_1_62754:
��
dense_1_62756:	�
z_mean_62759:	�
z_mean_62761:"
z_log_var_62764:	�
z_log_var_62766:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinput_maritalembedding_marital_62732*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_62211�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityembedding_ethnicity_62735*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_62225�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCall
input_raceembedding_race_62738*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_62239�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinput_genderembedding_gender_62741*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_62253�
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_62263�
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_62271�
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_62279�
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_62287�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0input_continuous*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_62299�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_62749dense_62751*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_62312�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_62754dense_1_62756*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_62329�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_62759z_mean_62761*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_62345�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_62764z_log_var_62766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_62361�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_62389v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 2>
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
:���������
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:���������
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:���������
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:���������
*
_user_specified_nameinput_continuous
�
�
'__inference_encoder_layer_call_fn_62897
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:	�

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_input_ethnicityinputs_input_genderinputs_input_maritalinputs_input_raceinputs_input_continuousunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_62394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:���������
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs/input_continuous
�
E
)__inference_flatten_3_layer_call_fn_63214

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_62287`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_embedding_ETHNICITY_layer_call_fn_63149

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_62225s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�C
�
B__inference_encoder_layer_call_and_return_conditional_losses_62821
input_ethnicity
input_gender
input_marital

input_race
input_continuous)
embedding_marital_62780:+
embedding_ethnicity_62783:&
embedding_race_62786:(
embedding_gender_62789:
dense_62797:	�
dense_62799:	�!
dense_1_62802:
��
dense_1_62804:	�
z_mean_62807:	�
z_mean_62809:"
z_log_var_62812:	�
z_log_var_62814:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinput_maritalembedding_marital_62780*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_62211�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityembedding_ethnicity_62783*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_62225�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCall
input_raceembedding_race_62786*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_62239�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinput_genderembedding_gender_62789*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_62253�
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_62263�
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_62271�
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_62279�
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_62287�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0input_continuous*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_62299�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_62797dense_62799*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_62312�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_62802dense_1_62804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_62329�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_62807z_mean_62809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_62345�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_62812z_log_var_62814*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_62361�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_62457v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 2>
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
:���������
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:���������
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:���������
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:���������
*
_user_specified_nameinput_continuous
�
k
<__inference_z_layer_call_and_return_conditional_losses_63351
inputs_0
inputs_1
identity�=
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
valueB:�
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
value	B :�
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
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2����
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
j
!__inference_z_layer_call_fn_63329
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_62457o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�	
�
+__inference_concatenate_layer_call_fn_63229
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_62299`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4
�
�
F__inference_concatenate_layer_call_and_return_conditional_losses_63239
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
value	B :�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������:���������:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/4
�m
�
 __inference__wrapped_model_62186
input_ethnicity
input_gender
input_marital

input_race
input_continuousB
0encoder_embedding_marital_embedding_lookup_62107:D
2encoder_embedding_ethnicity_embedding_lookup_62113:?
-encoder_embedding_race_embedding_lookup_62119:A
/encoder_embedding_gender_embedding_lookup_62125:?
,encoder_dense_matmul_readvariableop_resource:	�<
-encoder_dense_biasadd_readvariableop_resource:	�B
.encoder_dense_1_matmul_readvariableop_resource:
��>
/encoder_dense_1_biasadd_readvariableop_resource:	�@
-encoder_z_mean_matmul_readvariableop_resource:	�<
.encoder_z_mean_biasadd_readvariableop_resource:C
0encoder_z_log_var_matmul_readvariableop_resource:	�?
1encoder_z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2��$encoder/dense/BiasAdd/ReadVariableOp�#encoder/dense/MatMul/ReadVariableOp�&encoder/dense_1/BiasAdd/ReadVariableOp�%encoder/dense_1/MatMul/ReadVariableOp�,encoder/embedding_ETHNICITY/embedding_lookup�)encoder/embedding_GENDER/embedding_lookup�*encoder/embedding_MARITAL/embedding_lookup�'encoder/embedding_RACE/embedding_lookup�(encoder/z_log_var/BiasAdd/ReadVariableOp�'encoder/z_log_var/MatMul/ReadVariableOp�%encoder/z_mean/BiasAdd/ReadVariableOp�$encoder/z_mean/MatMul/ReadVariableOpv
encoder/embedding_MARITAL/CastCastinput_marital*

DstT0*

SrcT0*'
_output_shapes
:����������
*encoder/embedding_MARITAL/embedding_lookupResourceGather0encoder_embedding_marital_embedding_lookup_62107"encoder/embedding_MARITAL/Cast:y:0*
Tindices0*C
_class9
75loc:@encoder/embedding_MARITAL/embedding_lookup/62107*+
_output_shapes
:���������*
dtype0�
3encoder/embedding_MARITAL/embedding_lookup/IdentityIdentity3encoder/embedding_MARITAL/embedding_lookup:output:0*
T0*C
_class9
75loc:@encoder/embedding_MARITAL/embedding_lookup/62107*+
_output_shapes
:����������
5encoder/embedding_MARITAL/embedding_lookup/Identity_1Identity<encoder/embedding_MARITAL/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������z
 encoder/embedding_ETHNICITY/CastCastinput_ethnicity*

DstT0*

SrcT0*'
_output_shapes
:����������
,encoder/embedding_ETHNICITY/embedding_lookupResourceGather2encoder_embedding_ethnicity_embedding_lookup_62113$encoder/embedding_ETHNICITY/Cast:y:0*
Tindices0*E
_class;
97loc:@encoder/embedding_ETHNICITY/embedding_lookup/62113*+
_output_shapes
:���������*
dtype0�
5encoder/embedding_ETHNICITY/embedding_lookup/IdentityIdentity5encoder/embedding_ETHNICITY/embedding_lookup:output:0*
T0*E
_class;
97loc:@encoder/embedding_ETHNICITY/embedding_lookup/62113*+
_output_shapes
:����������
7encoder/embedding_ETHNICITY/embedding_lookup/Identity_1Identity>encoder/embedding_ETHNICITY/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������p
encoder/embedding_RACE/CastCast
input_race*

DstT0*

SrcT0*'
_output_shapes
:����������
'encoder/embedding_RACE/embedding_lookupResourceGather-encoder_embedding_race_embedding_lookup_62119encoder/embedding_RACE/Cast:y:0*
Tindices0*@
_class6
42loc:@encoder/embedding_RACE/embedding_lookup/62119*+
_output_shapes
:���������*
dtype0�
0encoder/embedding_RACE/embedding_lookup/IdentityIdentity0encoder/embedding_RACE/embedding_lookup:output:0*
T0*@
_class6
42loc:@encoder/embedding_RACE/embedding_lookup/62119*+
_output_shapes
:����������
2encoder/embedding_RACE/embedding_lookup/Identity_1Identity9encoder/embedding_RACE/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������t
encoder/embedding_GENDER/CastCastinput_gender*

DstT0*

SrcT0*'
_output_shapes
:����������
)encoder/embedding_GENDER/embedding_lookupResourceGather/encoder_embedding_gender_embedding_lookup_62125!encoder/embedding_GENDER/Cast:y:0*
Tindices0*B
_class8
64loc:@encoder/embedding_GENDER/embedding_lookup/62125*+
_output_shapes
:���������*
dtype0�
2encoder/embedding_GENDER/embedding_lookup/IdentityIdentity2encoder/embedding_GENDER/embedding_lookup:output:0*
T0*B
_class8
64loc:@encoder/embedding_GENDER/embedding_lookup/62125*+
_output_shapes
:����������
4encoder/embedding_GENDER/embedding_lookup/Identity_1Identity;encoder/embedding_GENDER/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������f
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
encoder/flatten/ReshapeReshape=encoder/embedding_GENDER/embedding_lookup/Identity_1:output:0encoder/flatten/Const:output:0*
T0*'
_output_shapes
:���������h
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
encoder/flatten_1/ReshapeReshape;encoder/embedding_RACE/embedding_lookup/Identity_1:output:0 encoder/flatten_1/Const:output:0*
T0*'
_output_shapes
:���������h
encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
encoder/flatten_2/ReshapeReshape@encoder/embedding_ETHNICITY/embedding_lookup/Identity_1:output:0 encoder/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������h
encoder/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
encoder/flatten_3/ReshapeReshape>encoder/embedding_MARITAL/embedding_lookup/Identity_1:output:0 encoder/flatten_3/Const:output:0*
T0*'
_output_shapes
:���������a
encoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
encoder/concatenate/concatConcatV2 encoder/flatten/Reshape:output:0"encoder/flatten_1/Reshape:output:0"encoder/flatten_2/Reshape:output:0"encoder/flatten_3/Reshape:output:0input_continuous(encoder/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
encoder/dense/MatMulMatMul#encoder/concatenate/concat:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
encoder/dense_1/MatMulMatMul encoder/dense/Relu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
encoder/dense_1/ReluRelu encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
encoder/z_mean/MatMulMatMul"encoder/dense_1/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
encoder/z_log_var/MatMulMatMul"encoder/dense_1/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������^
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
valueB:�
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
value	B :�
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
 *  �?�
,encoder/z/random_normal/RandomStandardNormalRandomStandardNormal&encoder/z/random_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2����
encoder/z/random_normal/mulMul5encoder/z/random_normal/RandomStandardNormal:output:0'encoder/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:����������
encoder/z/random_normalAddV2encoder/z/random_normal/mul:z:0%encoder/z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������T
encoder/z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
encoder/z/mulMulencoder/z/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������Y
encoder/z/ExpExpencoder/z/mul:z:0*
T0*'
_output_shapes
:���������x
encoder/z/mul_1Mulencoder/z/Exp:y:0encoder/z/random_normal:z:0*
T0*'
_output_shapes
:���������~
encoder/z/addAddV2encoder/z_mean/BiasAdd:output:0encoder/z/mul_1:z:0*
T0*'
_output_shapes
:���������`
IdentityIdentityencoder/z/add:z:0^NoOp*
T0*'
_output_shapes
:���������s

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������p

Identity_2Identityencoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp-^encoder/embedding_ETHNICITY/embedding_lookup*^encoder/embedding_GENDER/embedding_lookup+^encoder/embedding_MARITAL/embedding_lookup(^encoder/embedding_RACE/embedding_lookup)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 2L
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
:���������
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:���������
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:���������
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:���������
*
_user_specified_nameinput_continuous
�
�
1__inference_embedding_MARITAL_layer_call_fn_63166

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_62211s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_62860
input_ethnicity
input_gender
input_marital

input_race
input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:	�

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
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
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_62186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:���������
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:���������
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:���������
*
_user_specified_nameinput_continuous
�
�
&__inference_z_mean_layer_call_fn_63288

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_62345o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_63209

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�a
�

B__inference_encoder_layer_call_and_return_conditional_losses_63108
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous:
(embedding_marital_embedding_lookup_63029:<
*embedding_ethnicity_embedding_lookup_63035:7
%embedding_race_embedding_lookup_63041:9
'embedding_gender_embedding_lookup_63047:7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�8
%z_mean_matmul_readvariableop_resource:	�4
&z_mean_biasadd_readvariableop_resource:;
(z_log_var_matmul_readvariableop_resource:	�7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�$embedding_ETHNICITY/embedding_lookup�!embedding_GENDER/embedding_lookup�"embedding_MARITAL/embedding_lookup�embedding_RACE/embedding_lookup� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOpu
embedding_MARITAL/CastCastinputs_input_marital*

DstT0*

SrcT0*'
_output_shapes
:����������
"embedding_MARITAL/embedding_lookupResourceGather(embedding_marital_embedding_lookup_63029embedding_MARITAL/Cast:y:0*
Tindices0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/63029*+
_output_shapes
:���������*
dtype0�
+embedding_MARITAL/embedding_lookup/IdentityIdentity+embedding_MARITAL/embedding_lookup:output:0*
T0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/63029*+
_output_shapes
:����������
-embedding_MARITAL/embedding_lookup/Identity_1Identity4embedding_MARITAL/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������y
embedding_ETHNICITY/CastCastinputs_input_ethnicity*

DstT0*

SrcT0*'
_output_shapes
:����������
$embedding_ETHNICITY/embedding_lookupResourceGather*embedding_ethnicity_embedding_lookup_63035embedding_ETHNICITY/Cast:y:0*
Tindices0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/63035*+
_output_shapes
:���������*
dtype0�
-embedding_ETHNICITY/embedding_lookup/IdentityIdentity-embedding_ETHNICITY/embedding_lookup:output:0*
T0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/63035*+
_output_shapes
:����������
/embedding_ETHNICITY/embedding_lookup/Identity_1Identity6embedding_ETHNICITY/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������o
embedding_RACE/CastCastinputs_input_race*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_RACE/embedding_lookupResourceGather%embedding_race_embedding_lookup_63041embedding_RACE/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/63041*+
_output_shapes
:���������*
dtype0�
(embedding_RACE/embedding_lookup/IdentityIdentity(embedding_RACE/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/63041*+
_output_shapes
:����������
*embedding_RACE/embedding_lookup/Identity_1Identity1embedding_RACE/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������s
embedding_GENDER/CastCastinputs_input_gender*

DstT0*

SrcT0*'
_output_shapes
:����������
!embedding_GENDER/embedding_lookupResourceGather'embedding_gender_embedding_lookup_63047embedding_GENDER/Cast:y:0*
Tindices0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/63047*+
_output_shapes
:���������*
dtype0�
*embedding_GENDER/embedding_lookup/IdentityIdentity*embedding_GENDER/embedding_lookup:output:0*
T0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/63047*+
_output_shapes
:����������
,embedding_GENDER/embedding_lookup/Identity_1Identity3embedding_GENDER/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape5embedding_GENDER/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_1/ReshapeReshape3embedding_RACE/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:���������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_2/ReshapeReshape8embedding_ETHNICITY/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_3/ReshapeReshape6embedding_MARITAL/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0inputs_input_continuous concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
z_mean/MatMulMatMuldense_1/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
z_log_var/MatMulMatMuldense_1/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
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
valueB:�
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
value	B :�
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
 *  �?�
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2��M�
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:����������
z/random_normalAddV2z/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������L
z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
z/mulMulz/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������I
z/ExpExp	z/mul:z:0*
T0*'
_output_shapes
:���������`
z/mul_1Mul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:���������f
z/addAddV2z_mean/BiasAdd:output:0z/mul_1:z:0*
T0*'
_output_shapes
:���������f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	z/add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^embedding_ETHNICITY/embedding_lookup"^embedding_GENDER/embedding_lookup#^embedding_MARITAL/embedding_lookup ^embedding_RACE/embedding_lookup!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 2<
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
:���������
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs/input_continuous
�a
�

B__inference_encoder_layer_call_and_return_conditional_losses_63021
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous:
(embedding_marital_embedding_lookup_62942:<
*embedding_ethnicity_embedding_lookup_62948:7
%embedding_race_embedding_lookup_62954:9
'embedding_gender_embedding_lookup_62960:7
$dense_matmul_readvariableop_resource:	�4
%dense_biasadd_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�8
%z_mean_matmul_readvariableop_resource:	�4
&z_mean_biasadd_readvariableop_resource:;
(z_log_var_matmul_readvariableop_resource:	�7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�$embedding_ETHNICITY/embedding_lookup�!embedding_GENDER/embedding_lookup�"embedding_MARITAL/embedding_lookup�embedding_RACE/embedding_lookup� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOpu
embedding_MARITAL/CastCastinputs_input_marital*

DstT0*

SrcT0*'
_output_shapes
:����������
"embedding_MARITAL/embedding_lookupResourceGather(embedding_marital_embedding_lookup_62942embedding_MARITAL/Cast:y:0*
Tindices0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/62942*+
_output_shapes
:���������*
dtype0�
+embedding_MARITAL/embedding_lookup/IdentityIdentity+embedding_MARITAL/embedding_lookup:output:0*
T0*;
_class1
/-loc:@embedding_MARITAL/embedding_lookup/62942*+
_output_shapes
:����������
-embedding_MARITAL/embedding_lookup/Identity_1Identity4embedding_MARITAL/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������y
embedding_ETHNICITY/CastCastinputs_input_ethnicity*

DstT0*

SrcT0*'
_output_shapes
:����������
$embedding_ETHNICITY/embedding_lookupResourceGather*embedding_ethnicity_embedding_lookup_62948embedding_ETHNICITY/Cast:y:0*
Tindices0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/62948*+
_output_shapes
:���������*
dtype0�
-embedding_ETHNICITY/embedding_lookup/IdentityIdentity-embedding_ETHNICITY/embedding_lookup:output:0*
T0*=
_class3
1/loc:@embedding_ETHNICITY/embedding_lookup/62948*+
_output_shapes
:����������
/embedding_ETHNICITY/embedding_lookup/Identity_1Identity6embedding_ETHNICITY/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������o
embedding_RACE/CastCastinputs_input_race*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_RACE/embedding_lookupResourceGather%embedding_race_embedding_lookup_62954embedding_RACE/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/62954*+
_output_shapes
:���������*
dtype0�
(embedding_RACE/embedding_lookup/IdentityIdentity(embedding_RACE/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_RACE/embedding_lookup/62954*+
_output_shapes
:����������
*embedding_RACE/embedding_lookup/Identity_1Identity1embedding_RACE/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������s
embedding_GENDER/CastCastinputs_input_gender*

DstT0*

SrcT0*'
_output_shapes
:����������
!embedding_GENDER/embedding_lookupResourceGather'embedding_gender_embedding_lookup_62960embedding_GENDER/Cast:y:0*
Tindices0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/62960*+
_output_shapes
:���������*
dtype0�
*embedding_GENDER/embedding_lookup/IdentityIdentity*embedding_GENDER/embedding_lookup:output:0*
T0*:
_class0
.,loc:@embedding_GENDER/embedding_lookup/62960*+
_output_shapes
:����������
,embedding_GENDER/embedding_lookup/Identity_1Identity3embedding_GENDER/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape5embedding_GENDER/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_1/ReshapeReshape3embedding_RACE/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:���������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_2/ReshapeReshape8embedding_ETHNICITY/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten_3/ReshapeReshape6embedding_MARITAL/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:���������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0inputs_input_continuous concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
z_mean/MatMulMatMuldense_1/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
z_log_var/MatMulMatMuldense_1/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
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
valueB:�
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
value	B :�
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
 *  �?�
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2���
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:����������
z/random_normalAddV2z/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:���������L
z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
z/mulMulz/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������I
z/ExpExp	z/mul:z:0*
T0*'
_output_shapes
:���������`
z/mul_1Mul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:���������f
z/addAddV2z_mean/BiasAdd:output:0z/mul_1:z:0*
T0*'
_output_shapes
:���������f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	z/add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^embedding_ETHNICITY/embedding_lookup"^embedding_GENDER/embedding_lookup#^embedding_MARITAL/embedding_lookup ^embedding_RACE/embedding_lookup!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 2<
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
:���������
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs/input_continuous
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_63220

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_63279

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_63142

inputs(
embedding_lookup_63136:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_63136Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/63136*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/63136*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
A__inference_z_mean_layer_call_and_return_conditional_losses_62345

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_63176

inputs(
embedding_lookup_63170:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_63170Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/63170*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/63170*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_62934
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:	�

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_input_ethnicityinputs_input_genderinputs_input_maritalinputs_input_raceinputs_input_continuousunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_62657o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:���������
0
_user_specified_nameinputs/input_ETHNICITY:\X
'
_output_shapes
:���������
-
_user_specified_nameinputs/input_GENDER:]Y
'
_output_shapes
:���������
.
_user_specified_nameinputs/input_MARITAL:ZV
'
_output_shapes
:���������
+
_user_specified_nameinputs/input_RACE:`\
'
_output_shapes
:���������
1
_user_specified_nameinputs/input_continuous
�
j
!__inference_z_layer_call_fn_63323
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_62389o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�	
�
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_62225

inputs(
embedding_lookup_62219:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_62219Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/62219*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/62219*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_62279

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_62239

inputs(
embedding_lookup_62233:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_62233Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/62233*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/62233*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_63268

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_62329p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_62253

inputs(
embedding_lookup_62247:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_62247Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/62247*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/62247*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_63259

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_62425
input_ethnicity
input_gender
input_marital

input_race
input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:	�

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
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
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_62394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:���������
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:���������
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:���������
*
_user_specified_nameinput_continuous
�B
�
B__inference_encoder_layer_call_and_return_conditional_losses_62394

inputs
inputs_1
inputs_2
inputs_3
inputs_4)
embedding_marital_62212:+
embedding_ethnicity_62226:&
embedding_race_62240:(
embedding_gender_62254:
dense_62313:	�
dense_62315:	�!
dense_1_62330:
��
dense_1_62332:	�
z_mean_62346:	�
z_mean_62348:"
z_log_var_62362:	�
z_log_var_62364:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_marital_62212*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_62211�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_ethnicity_62226*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_62225�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_race_62240*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_62239�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_gender_62254*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_62253�
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_62263�
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_62271�
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_62279�
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_62287�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_62299�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_62313dense_62315*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_62312�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_62330dense_1_62332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_62329�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_62346z_mean_62348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_62345�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_62362z_log_var_62364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_62361�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_62389v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 2>
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
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_63248

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_62312p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_62287

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_embedding_RACE_layer_call_fn_63132

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_62239s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_62312

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�B
�
B__inference_encoder_layer_call_and_return_conditional_losses_62657

inputs
inputs_1
inputs_2
inputs_3
inputs_4)
embedding_marital_62616:+
embedding_ethnicity_62619:&
embedding_race_62622:(
embedding_gender_62625:
dense_62633:	�
dense_62635:	�!
dense_1_62638:
��
dense_1_62640:	�
z_mean_62643:	�
z_mean_62645:"
z_log_var_62648:	�
z_log_var_62650:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_marital_62616*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_62211�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_ethnicity_62619*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_62225�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_race_62622*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_62239�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_gender_62625*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_62253�
flatten/PartitionedCallPartitionedCall1embedding_GENDER/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_62263�
flatten_1/PartitionedCallPartitionedCall/embedding_RACE/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_62271�
flatten_2/PartitionedCallPartitionedCall4embedding_ETHNICITY/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_62279�
flatten_3/PartitionedCallPartitionedCall2embedding_MARITAL/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_62287�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_62299�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_62633dense_62635*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_62312�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_62638dense_1_62640*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_62329�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_62643z_mean_62645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_62345�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_62648z_log_var_62650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_62361�
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_62457v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������s

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall,^embedding_ETHNICITY/StatefulPartitionedCall)^embedding_GENDER/StatefulPartitionedCall*^embedding_MARITAL/StatefulPartitionedCall'^embedding_RACE/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 2>
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
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_63198

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
D__inference_z_log_var_layer_call_and_return_conditional_losses_62361

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_62329

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_63125

inputs(
embedding_lookup_63119:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_63119Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/63119*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/63119*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
i
<__inference_z_layer_call_and_return_conditional_losses_62389

inputs
inputs_1
identity�;
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
valueB:�
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
value	B :�
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
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2����
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_62211

inputs(
embedding_lookup_62205:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_62205Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/62205*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/62205*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
__inference__traced_save_63438
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

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_embedding_gender_embeddings_read_readvariableop4savev2_embedding_race_embeddings_read_readvariableop9savev2_embedding_ethnicity_embeddings_read_readvariableop7savev2_embedding_marital_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapesu
s: :::::	�:�:
��:�:	�::	�:: 2(
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
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%	!

_output_shapes
:	�: 


_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�	
�
D__inference_z_log_var_layer_call_and_return_conditional_losses_63317

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_flatten_1_layer_call_fn_63192

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_62271`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�3
�
!__inference__traced_restore_63484
file_prefix>
,assignvariableop_embedding_gender_embeddings:>
,assignvariableop_1_embedding_race_embeddings:C
1assignvariableop_2_embedding_ethnicity_embeddings:A
/assignvariableop_3_embedding_marital_embeddings:2
assignvariableop_4_dense_kernel:	�,
assignvariableop_5_dense_bias:	�5
!assignvariableop_6_dense_1_kernel:
��.
assignvariableop_7_dense_1_bias:	�3
 assignvariableop_8_z_mean_kernel:	�,
assignvariableop_9_z_mean_bias:7
$assignvariableop_10_z_log_var_kernel:	�0
"assignvariableop_11_z_log_var_bias:
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp,assignvariableop_embedding_gender_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp,assignvariableop_1_embedding_race_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp1assignvariableop_2_embedding_ethnicity_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_embedding_marital_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_z_mean_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_z_mean_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_z_log_var_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_z_log_var_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: �
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
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_63187

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_63159

inputs(
embedding_lookup_63153:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_63153Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/63153*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/63153*+
_output_shapes
:����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_concatenate_layer_call_and_return_conditional_losses_62299

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
value	B :�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:���������:���������:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_62725
input_ethnicity
input_gender
input_marital

input_race
input_continuous
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:	�

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
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
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_62657o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesy
w:���������:���������:���������:���������:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_nameinput_ETHNICITY:UQ
'
_output_shapes
:���������
&
_user_specified_nameinput_GENDER:VR
'
_output_shapes
:���������
'
_user_specified_nameinput_MARITAL:SO
'
_output_shapes
:���������
$
_user_specified_name
input_RACE:YU
'
_output_shapes
:���������
*
_user_specified_nameinput_continuous
�
k
<__inference_z_layer_call_and_return_conditional_losses_63373
inputs_0
inputs_1
identity�=
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
valueB:�
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
value	B :�
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
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*

seed**
seed2����
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
C
'__inference_flatten_layer_call_fn_63181

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_62263`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_z_log_var_layer_call_fn_63307

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_62361o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_62271

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_embedding_GENDER_layer_call_fn_63115

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_62253s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
input_ETHNICITY8
!serving_default_input_ETHNICITY:0���������
E
input_GENDER5
serving_default_input_GENDER:0���������
G
input_MARITAL6
serving_default_input_MARITAL:0���������
A

input_RACE3
serving_default_input_RACE:0���������
M
input_continuous9
"serving_default_input_continuous:0���������5
z0
StatefulPartitionedCall:0���������=
	z_log_var0
StatefulPartitionedCall:1���������:
z_mean0
StatefulPartitionedCall:2���������tensorflow/serving/predict:��
�
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
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"
embeddings"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
embeddings"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0
embeddings"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7
embeddings"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias"
_tf_keras_layer
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias"
_tf_keras_layer
�
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
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
'__inference_encoder_layer_call_fn_62425
'__inference_encoder_layer_call_fn_62897
'__inference_encoder_layer_call_fn_62934
'__inference_encoder_layer_call_fn_62725�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
B__inference_encoder_layer_call_and_return_conditional_losses_63021
B__inference_encoder_layer_call_and_return_conditional_losses_63108
B__inference_encoder_layer_call_and_return_conditional_losses_62773
B__inference_encoder_layer_call_and_return_conditional_losses_62821�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_62186input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_embedding_GENDER_layer_call_fn_63115�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_63125�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+2embedding_GENDER/embeddings
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_embedding_RACE_layer_call_fn_63132�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_63142�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)2embedding_RACE/embeddings
'
00"
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_embedding_ETHNICITY_layer_call_fn_63149�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_63159�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0:.2embedding_ETHNICITY/embeddings
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_embedding_MARITAL_layer_call_fn_63166�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_63176�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,2embedding_MARITAL/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_63181�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_63187�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_1_layer_call_fn_63192�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_1_layer_call_and_return_conditional_losses_63198�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_2_layer_call_fn_63203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_2_layer_call_and_return_conditional_losses_63209�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_3_layer_call_fn_63214�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_flatten_3_layer_call_and_return_conditional_losses_63220�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_concatenate_layer_call_fn_63229�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_concatenate_layer_call_and_return_conditional_losses_63239�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense_layer_call_fn_63248�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_63259�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2dense/kernel
:�2
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_63268�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_63279�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2dense_1/kernel
:�2dense_1/bias
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_z_mean_layer_call_fn_63288�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_z_mean_layer_call_and_return_conditional_losses_63298�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :	�2z_mean/kernel
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_z_log_var_layer_call_fn_63307�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_z_log_var_layer_call_and_return_conditional_losses_63317�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!	�2z_log_var/kernel
:2z_log_var/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
!__inference_z_layer_call_fn_63323
!__inference_z_layer_call_fn_63329�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
<__inference_z_layer_call_and_return_conditional_losses_63351
<__inference_z_layer_call_and_return_conditional_losses_63373�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
�
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
�B�
'__inference_encoder_layer_call_fn_62425input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_encoder_layer_call_fn_62897inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_encoder_layer_call_fn_62934inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_encoder_layer_call_fn_62725input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_encoder_layer_call_and_return_conditional_losses_63021inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_encoder_layer_call_and_return_conditional_losses_63108inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_encoder_layer_call_and_return_conditional_losses_62773input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_encoder_layer_call_and_return_conditional_losses_62821input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_62860input_ETHNICITYinput_GENDERinput_MARITAL
input_RACEinput_continuous"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_embedding_GENDER_layer_call_fn_63115inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_63125inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_embedding_RACE_layer_call_fn_63132inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_63142inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
3__inference_embedding_ETHNICITY_layer_call_fn_63149inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_63159inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_embedding_MARITAL_layer_call_fn_63166inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_63176inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_flatten_layer_call_fn_63181inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_63187inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_flatten_1_layer_call_fn_63192inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_1_layer_call_and_return_conditional_losses_63198inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_flatten_2_layer_call_fn_63203inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_2_layer_call_and_return_conditional_losses_63209inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_flatten_3_layer_call_fn_63214inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_flatten_3_layer_call_and_return_conditional_losses_63220inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_concatenate_layer_call_fn_63229inputs/0inputs/1inputs/2inputs/3inputs/4"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_concatenate_layer_call_and_return_conditional_losses_63239inputs/0inputs/1inputs/2inputs/3inputs/4"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
%__inference_dense_layer_call_fn_63248inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_63259inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
'__inference_dense_1_layer_call_fn_63268inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_63279inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
&__inference_z_mean_layer_call_fn_63288inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_z_mean_layer_call_and_return_conditional_losses_63298inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_z_log_var_layer_call_fn_63307inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_z_log_var_layer_call_and_return_conditional_losses_63317inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
!__inference_z_layer_call_fn_63323inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
!__inference_z_layer_call_fn_63329inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_z_layer_call_and_return_conditional_losses_63351inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
<__inference_z_layer_call_and_return_conditional_losses_63373inputs/0inputs/1"�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_62186�70)"\]delmtu���
���
���
<
input_ETHNICITY)�&
input_ETHNICITY���������
6
input_GENDER&�#
input_GENDER���������
8
input_MARITAL'�$
input_MARITAL���������
2

input_RACE$�!

input_RACE���������
>
input_continuous*�'
input_continuous���������
� "���
 
z�
z���������
0
	z_log_var#� 
	z_log_var���������
*
z_mean �
z_mean����������
F__inference_concatenate_layer_call_and_return_conditional_losses_63239����
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
� "%�"
�
0���������
� �
+__inference_concatenate_layer_call_fn_63229����
���
���
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
"�
inputs/3���������
"�
inputs/4���������
� "�����������
B__inference_dense_1_layer_call_and_return_conditional_losses_63279^de0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dense_1_layer_call_fn_63268Qde0�-
&�#
!�
inputs����������
� "������������
@__inference_dense_layer_call_and_return_conditional_losses_63259]\]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� y
%__inference_dense_layer_call_fn_63248P\]/�,
%�"
 �
inputs���������
� "������������
N__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_63159_0/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
3__inference_embedding_ETHNICITY_layer_call_fn_63149R0/�,
%�"
 �
inputs���������
� "�����������
K__inference_embedding_GENDER_layer_call_and_return_conditional_losses_63125_"/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
0__inference_embedding_GENDER_layer_call_fn_63115R"/�,
%�"
 �
inputs���������
� "�����������
L__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_63176_7/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
1__inference_embedding_MARITAL_layer_call_fn_63166R7/�,
%�"
 �
inputs���������
� "�����������
I__inference_embedding_RACE_layer_call_and_return_conditional_losses_63142_)/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
.__inference_embedding_RACE_layer_call_fn_63132R)/�,
%�"
 �
inputs���������
� "�����������
B__inference_encoder_layer_call_and_return_conditional_losses_62773�70)"\]delmtu���
���
���
<
input_ETHNICITY)�&
input_ETHNICITY���������
6
input_GENDER&�#
input_GENDER���������
8
input_MARITAL'�$
input_MARITAL���������
2

input_RACE$�!

input_RACE���������
>
input_continuous*�'
input_continuous���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_62821�70)"\]delmtu���
���
���
<
input_ETHNICITY)�&
input_ETHNICITY���������
6
input_GENDER&�#
input_GENDER���������
8
input_MARITAL'�$
input_MARITAL���������
2

input_RACE$�!

input_RACE���������
>
input_continuous*�'
input_continuous���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_63021�70)"\]delmtu���
���
���
C
input_ETHNICITY0�-
inputs/input_ETHNICITY���������
=
input_GENDER-�*
inputs/input_GENDER���������
?
input_MARITAL.�+
inputs/input_MARITAL���������
9

input_RACE+�(
inputs/input_RACE���������
E
input_continuous1�.
inputs/input_continuous���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_63108�70)"\]delmtu���
���
���
C
input_ETHNICITY0�-
inputs/input_ETHNICITY���������
=
input_GENDER-�*
inputs/input_GENDER���������
?
input_MARITAL.�+
inputs/input_MARITAL���������
9

input_RACE+�(
inputs/input_RACE���������
E
input_continuous1�.
inputs/input_continuous���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
'__inference_encoder_layer_call_fn_62425�70)"\]delmtu���
���
���
<
input_ETHNICITY)�&
input_ETHNICITY���������
6
input_GENDER&�#
input_GENDER���������
8
input_MARITAL'�$
input_MARITAL���������
2

input_RACE$�!

input_RACE���������
>
input_continuous*�'
input_continuous���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_encoder_layer_call_fn_62725�70)"\]delmtu���
���
���
<
input_ETHNICITY)�&
input_ETHNICITY���������
6
input_GENDER&�#
input_GENDER���������
8
input_MARITAL'�$
input_MARITAL���������
2

input_RACE$�!

input_RACE���������
>
input_continuous*�'
input_continuous���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_encoder_layer_call_fn_62897�70)"\]delmtu���
���
���
C
input_ETHNICITY0�-
inputs/input_ETHNICITY���������
=
input_GENDER-�*
inputs/input_GENDER���������
?
input_MARITAL.�+
inputs/input_MARITAL���������
9

input_RACE+�(
inputs/input_RACE���������
E
input_continuous1�.
inputs/input_continuous���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_encoder_layer_call_fn_62934�70)"\]delmtu���
���
���
C
input_ETHNICITY0�-
inputs/input_ETHNICITY���������
=
input_GENDER-�*
inputs/input_GENDER���������
?
input_MARITAL.�+
inputs/input_MARITAL���������
9

input_RACE+�(
inputs/input_RACE���������
E
input_continuous1�.
inputs/input_continuous���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
D__inference_flatten_1_layer_call_and_return_conditional_losses_63198\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� |
)__inference_flatten_1_layer_call_fn_63192O3�0
)�&
$�!
inputs���������
� "�����������
D__inference_flatten_2_layer_call_and_return_conditional_losses_63209\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� |
)__inference_flatten_2_layer_call_fn_63203O3�0
)�&
$�!
inputs���������
� "�����������
D__inference_flatten_3_layer_call_and_return_conditional_losses_63220\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� |
)__inference_flatten_3_layer_call_fn_63214O3�0
)�&
$�!
inputs���������
� "�����������
B__inference_flatten_layer_call_and_return_conditional_losses_63187\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� z
'__inference_flatten_layer_call_fn_63181O3�0
)�&
$�!
inputs���������
� "�����������
#__inference_signature_wrapper_62860�70)"\]delmtu���
� 
���
<
input_ETHNICITY)�&
input_ETHNICITY���������
6
input_GENDER&�#
input_GENDER���������
8
input_MARITAL'�$
input_MARITAL���������
2

input_RACE$�!

input_RACE���������
>
input_continuous*�'
input_continuous���������"���
 
z�
z���������
0
	z_log_var#� 
	z_log_var���������
*
z_mean �
z_mean����������
<__inference_z_layer_call_and_return_conditional_losses_63351�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p 
� "%�"
�
0���������
� �
<__inference_z_layer_call_and_return_conditional_losses_63373�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p
� "%�"
�
0���������
� �
!__inference_z_layer_call_fn_63323~b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p 
� "�����������
!__inference_z_layer_call_fn_63329~b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p
� "�����������
D__inference_z_log_var_layer_call_and_return_conditional_losses_63317]tu0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_z_log_var_layer_call_fn_63307Ptu0�-
&�#
!�
inputs����������
� "�����������
A__inference_z_mean_layer_call_and_return_conditional_losses_63298]lm0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� z
&__inference_z_mean_layer_call_fn_63288Plm0�-
&�#
!�
inputs����������
� "����������