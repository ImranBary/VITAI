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
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_220914

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
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_221492
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
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_221538��
�
�
G__inference_concatenate_layer_call_and_return_conditional_losses_221293
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
�	
�
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_220279

inputs)
embedding_lookup_220273:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_220273Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/220273*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/220273*+
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
,__inference_concatenate_layer_call_fn_221283
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
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_220353`
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
�
j
=__inference_z_layer_call_and_return_conditional_losses_220511

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
�3
�
"__inference__traced_restore_221538
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
�
�
*__inference_z_log_var_layer_call_fn_221361

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
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_220415o
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
�	
�
B__inference_z_mean_layer_call_and_return_conditional_losses_220399

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
�
�
(__inference_encoder_layer_call_fn_220988
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
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_220711o
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
�	
�
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_220265

inputs)
embedding_lookup_220259:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_220259Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/220259*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/220259*+
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
�
�
(__inference_encoder_layer_call_fn_220479
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
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_220448o
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
�	
�
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_221179

inputs)
embedding_lookup_221173:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_221173Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/221173*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/221173*+
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
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_221230

inputs)
embedding_lookup_221224:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_221224Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/221224*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/221224*+
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
�
F
*__inference_flatten_3_layer_call_fn_221268

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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_220341`
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
k
"__inference_z_layer_call_fn_221377
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
 *0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_220443o
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
�B
�
C__inference_encoder_layer_call_and_return_conditional_losses_220711

inputs
inputs_1
inputs_2
inputs_3
inputs_4*
embedding_marital_220670:,
embedding_ethnicity_220673:'
embedding_race_220676:)
embedding_gender_220679:
dense_220687:	�
dense_220689:	�"
dense_1_220692:
��
dense_1_220694:	� 
z_mean_220697:	�
z_mean_220699:#
z_log_var_220702:	�
z_log_var_220704:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_marital_220670*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_220265�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_ethnicity_220673*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_220279�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_race_220676*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_220293�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_gender_220679*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_220307�
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
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_220317�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_220325�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_220333�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_220341�
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
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_220353�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220687dense_220689*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220366�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_220692dense_1_220694*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220383�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_220697z_mean_220699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_220399�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_220702z_log_var_220704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_220415�
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
 *0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_220511v
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
�a
�

C__inference_encoder_layer_call_and_return_conditional_losses_221075
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous;
)embedding_marital_embedding_lookup_220996:=
+embedding_ethnicity_embedding_lookup_221002:8
&embedding_race_embedding_lookup_221008::
(embedding_gender_embedding_lookup_221014:7
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
"embedding_MARITAL/embedding_lookupResourceGather)embedding_marital_embedding_lookup_220996embedding_MARITAL/Cast:y:0*
Tindices0*<
_class2
0.loc:@embedding_MARITAL/embedding_lookup/220996*+
_output_shapes
:���������*
dtype0�
+embedding_MARITAL/embedding_lookup/IdentityIdentity+embedding_MARITAL/embedding_lookup:output:0*
T0*<
_class2
0.loc:@embedding_MARITAL/embedding_lookup/220996*+
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
$embedding_ETHNICITY/embedding_lookupResourceGather+embedding_ethnicity_embedding_lookup_221002embedding_ETHNICITY/Cast:y:0*
Tindices0*>
_class4
20loc:@embedding_ETHNICITY/embedding_lookup/221002*+
_output_shapes
:���������*
dtype0�
-embedding_ETHNICITY/embedding_lookup/IdentityIdentity-embedding_ETHNICITY/embedding_lookup:output:0*
T0*>
_class4
20loc:@embedding_ETHNICITY/embedding_lookup/221002*+
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
embedding_RACE/embedding_lookupResourceGather&embedding_race_embedding_lookup_221008embedding_RACE/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_RACE/embedding_lookup/221008*+
_output_shapes
:���������*
dtype0�
(embedding_RACE/embedding_lookup/IdentityIdentity(embedding_RACE/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_RACE/embedding_lookup/221008*+
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
!embedding_GENDER/embedding_lookupResourceGather(embedding_gender_embedding_lookup_221014embedding_GENDER/Cast:y:0*
Tindices0*;
_class1
/-loc:@embedding_GENDER/embedding_lookup/221014*+
_output_shapes
:���������*
dtype0�
*embedding_GENDER/embedding_lookup/IdentityIdentity*embedding_GENDER/embedding_lookup:output:0*
T0*;
_class1
/-loc:@embedding_GENDER/embedding_lookup/221014*+
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
�
�
2__inference_embedding_MARITAL_layer_call_fn_221220

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
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_220265s
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
B__inference_z_mean_layer_call_and_return_conditional_losses_221352

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
�
�
&__inference_dense_layer_call_fn_221302

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
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220366p
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
�C
�
C__inference_encoder_layer_call_and_return_conditional_losses_220827
input_ethnicity
input_gender
input_marital

input_race
input_continuous*
embedding_marital_220786:,
embedding_ethnicity_220789:'
embedding_race_220792:)
embedding_gender_220795:
dense_220803:	�
dense_220805:	�"
dense_1_220808:
��
dense_1_220810:	� 
z_mean_220813:	�
z_mean_220815:#
z_log_var_220818:	�
z_log_var_220820:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinput_maritalembedding_marital_220786*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_220265�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityembedding_ethnicity_220789*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_220279�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCall
input_raceembedding_race_220792*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_220293�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinput_genderembedding_gender_220795*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_220307�
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
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_220317�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_220325�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_220333�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_220341�
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
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_220353�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220803dense_220805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220366�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_220808dense_1_220810*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220383�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_220813z_mean_220815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_220399�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_220818z_log_var_220820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_220415�
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
 *0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_220443v
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
�C
�
C__inference_encoder_layer_call_and_return_conditional_losses_220875
input_ethnicity
input_gender
input_marital

input_race
input_continuous*
embedding_marital_220834:,
embedding_ethnicity_220837:'
embedding_race_220840:)
embedding_gender_220843:
dense_220851:	�
dense_220853:	�"
dense_1_220856:
��
dense_1_220858:	� 
z_mean_220861:	�
z_mean_220863:#
z_log_var_220866:	�
z_log_var_220868:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinput_maritalembedding_marital_220834*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_220265�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinput_ethnicityembedding_ethnicity_220837*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_220279�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCall
input_raceembedding_race_220840*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_220293�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinput_genderembedding_gender_220843*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_220307�
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
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_220317�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_220325�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_220333�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_220341�
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
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_220353�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220851dense_220853*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220366�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_220856dense_1_220858*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220383�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_220861z_mean_220863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_220399�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_220866z_log_var_220868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_220415�
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
 *0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_220511v
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
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_220325

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
E__inference_z_log_var_layer_call_and_return_conditional_losses_221371

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
�m
�
!__inference__wrapped_model_220240
input_ethnicity
input_gender
input_marital

input_race
input_continuousC
1encoder_embedding_marital_embedding_lookup_220161:E
3encoder_embedding_ethnicity_embedding_lookup_220167:@
.encoder_embedding_race_embedding_lookup_220173:B
0encoder_embedding_gender_embedding_lookup_220179:?
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
*encoder/embedding_MARITAL/embedding_lookupResourceGather1encoder_embedding_marital_embedding_lookup_220161"encoder/embedding_MARITAL/Cast:y:0*
Tindices0*D
_class:
86loc:@encoder/embedding_MARITAL/embedding_lookup/220161*+
_output_shapes
:���������*
dtype0�
3encoder/embedding_MARITAL/embedding_lookup/IdentityIdentity3encoder/embedding_MARITAL/embedding_lookup:output:0*
T0*D
_class:
86loc:@encoder/embedding_MARITAL/embedding_lookup/220161*+
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
,encoder/embedding_ETHNICITY/embedding_lookupResourceGather3encoder_embedding_ethnicity_embedding_lookup_220167$encoder/embedding_ETHNICITY/Cast:y:0*
Tindices0*F
_class<
:8loc:@encoder/embedding_ETHNICITY/embedding_lookup/220167*+
_output_shapes
:���������*
dtype0�
5encoder/embedding_ETHNICITY/embedding_lookup/IdentityIdentity5encoder/embedding_ETHNICITY/embedding_lookup:output:0*
T0*F
_class<
:8loc:@encoder/embedding_ETHNICITY/embedding_lookup/220167*+
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
'encoder/embedding_RACE/embedding_lookupResourceGather.encoder_embedding_race_embedding_lookup_220173encoder/embedding_RACE/Cast:y:0*
Tindices0*A
_class7
53loc:@encoder/embedding_RACE/embedding_lookup/220173*+
_output_shapes
:���������*
dtype0�
0encoder/embedding_RACE/embedding_lookup/IdentityIdentity0encoder/embedding_RACE/embedding_lookup:output:0*
T0*A
_class7
53loc:@encoder/embedding_RACE/embedding_lookup/220173*+
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
)encoder/embedding_GENDER/embedding_lookupResourceGather0encoder_embedding_gender_embedding_lookup_220179!encoder/embedding_GENDER/Cast:y:0*
Tindices0*C
_class9
75loc:@encoder/embedding_GENDER/embedding_lookup/220179*+
_output_shapes
:���������*
dtype0�
2encoder/embedding_GENDER/embedding_lookup/IdentityIdentity2encoder/embedding_GENDER/embedding_lookup:output:0*
T0*C
_class9
75loc:@encoder/embedding_GENDER/embedding_lookup/220179*+
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
�B
�
C__inference_encoder_layer_call_and_return_conditional_losses_220448

inputs
inputs_1
inputs_2
inputs_3
inputs_4*
embedding_marital_220266:,
embedding_ethnicity_220280:'
embedding_race_220294:)
embedding_gender_220308:
dense_220367:	�
dense_220369:	�"
dense_1_220384:
��
dense_1_220386:	� 
z_mean_220400:	�
z_mean_220402:#
z_log_var_220416:	�
z_log_var_220418:
identity

identity_1

identity_2��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�+embedding_ETHNICITY/StatefulPartitionedCall�(embedding_GENDER/StatefulPartitionedCall�)embedding_MARITAL/StatefulPartitionedCall�&embedding_RACE/StatefulPartitionedCall�z/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
)embedding_MARITAL/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_marital_220266*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_220265�
+embedding_ETHNICITY/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_ethnicity_220280*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_220279�
&embedding_RACE/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_race_220294*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_220293�
(embedding_GENDER/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_gender_220308*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_220307�
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
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_220317�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_220325�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_220333�
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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_220341�
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
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_220353�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_220367dense_220369*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_220366�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_220384dense_1_220386*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220383�
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_220400z_mean_220402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_220399�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_220416z_log_var_220418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_220415�
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
 *0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_220443v
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
_
C__inference_flatten_layer_call_and_return_conditional_losses_220317

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
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_220293

inputs)
embedding_lookup_220287:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_220287Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/220287*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/220287*+
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
�
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_221274

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
�
�
(__inference_encoder_layer_call_fn_220951
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
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_220448o
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
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_220383

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
�
�
4__inference_embedding_ETHNICITY_layer_call_fn_221203

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
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_220279s
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
�	
�
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_221196

inputs)
embedding_lookup_221190:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_221190Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/221190*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/221190*+
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
__inference__traced_save_221492
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
�
k
"__inference_z_layer_call_fn_221383
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
 *0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_220511o
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
C__inference_dense_1_layer_call_and_return_conditional_losses_221333

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
�
F
*__inference_flatten_2_layer_call_fn_221257

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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_220333`
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
�
l
=__inference_z_layer_call_and_return_conditional_losses_221405
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
�	
�
E__inference_z_log_var_layer_call_and_return_conditional_losses_220415

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
�
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_220341

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
�
�
G__inference_concatenate_layer_call_and_return_conditional_losses_220353

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
�
D
(__inference_flatten_layer_call_fn_221235

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
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_220317`
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
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_220333

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
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_221241

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
�
�
/__inference_embedding_RACE_layer_call_fn_221186

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
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_220293s
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
(__inference_encoder_layer_call_fn_220779
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
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_220711o
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
1__inference_embedding_GENDER_layer_call_fn_221169

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
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_220307s
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
�

�
A__inference_dense_layer_call_and_return_conditional_losses_221313

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
l
=__inference_z_layer_call_and_return_conditional_losses_221427
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
�

�
A__inference_dense_layer_call_and_return_conditional_losses_220366

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
�a
�

C__inference_encoder_layer_call_and_return_conditional_losses_221162
inputs_input_ethnicity
inputs_input_gender
inputs_input_marital
inputs_input_race
inputs_input_continuous;
)embedding_marital_embedding_lookup_221083:=
+embedding_ethnicity_embedding_lookup_221089:8
&embedding_race_embedding_lookup_221095::
(embedding_gender_embedding_lookup_221101:7
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
"embedding_MARITAL/embedding_lookupResourceGather)embedding_marital_embedding_lookup_221083embedding_MARITAL/Cast:y:0*
Tindices0*<
_class2
0.loc:@embedding_MARITAL/embedding_lookup/221083*+
_output_shapes
:���������*
dtype0�
+embedding_MARITAL/embedding_lookup/IdentityIdentity+embedding_MARITAL/embedding_lookup:output:0*
T0*<
_class2
0.loc:@embedding_MARITAL/embedding_lookup/221083*+
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
$embedding_ETHNICITY/embedding_lookupResourceGather+embedding_ethnicity_embedding_lookup_221089embedding_ETHNICITY/Cast:y:0*
Tindices0*>
_class4
20loc:@embedding_ETHNICITY/embedding_lookup/221089*+
_output_shapes
:���������*
dtype0�
-embedding_ETHNICITY/embedding_lookup/IdentityIdentity-embedding_ETHNICITY/embedding_lookup:output:0*
T0*>
_class4
20loc:@embedding_ETHNICITY/embedding_lookup/221089*+
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
embedding_RACE/embedding_lookupResourceGather&embedding_race_embedding_lookup_221095embedding_RACE/Cast:y:0*
Tindices0*9
_class/
-+loc:@embedding_RACE/embedding_lookup/221095*+
_output_shapes
:���������*
dtype0�
(embedding_RACE/embedding_lookup/IdentityIdentity(embedding_RACE/embedding_lookup:output:0*
T0*9
_class/
-+loc:@embedding_RACE/embedding_lookup/221095*+
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
!embedding_GENDER/embedding_lookupResourceGather(embedding_gender_embedding_lookup_221101embedding_GENDER/Cast:y:0*
Tindices0*;
_class1
/-loc:@embedding_GENDER/embedding_lookup/221101*+
_output_shapes
:���������*
dtype0�
*embedding_GENDER/embedding_lookup/IdentityIdentity*embedding_GENDER/embedding_lookup:output:0*
T0*;
_class1
/-loc:@embedding_GENDER/embedding_lookup/221101*+
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
�
�
$__inference_signature_wrapper_220914
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
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_220240o
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
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_221252

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
�
F
*__inference_flatten_1_layer_call_fn_221246

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
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_220325`
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
(__inference_dense_1_layer_call_fn_221322

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
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_220383p
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
�
�
'__inference_z_mean_layer_call_fn_221342

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
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_220399o
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
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_221263

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
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_221213

inputs)
embedding_lookup_221207:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_221207Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/221207*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/221207*+
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
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_220307

inputs)
embedding_lookup_220301:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_220301Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/220301*+
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/220301*+
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
j
=__inference_z_layer_call_and_return_conditional_losses_220443

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
(__inference_encoder_layer_call_fn_220479
(__inference_encoder_layer_call_fn_220951
(__inference_encoder_layer_call_fn_220988
(__inference_encoder_layer_call_fn_220779�
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
C__inference_encoder_layer_call_and_return_conditional_losses_221075
C__inference_encoder_layer_call_and_return_conditional_losses_221162
C__inference_encoder_layer_call_and_return_conditional_losses_220827
C__inference_encoder_layer_call_and_return_conditional_losses_220875�
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
!__inference__wrapped_model_220240input_ETHNICITYinput_GENDERinput_MARITAL
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
1__inference_embedding_GENDER_layer_call_fn_221169�
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
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_221179�
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
/__inference_embedding_RACE_layer_call_fn_221186�
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
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_221196�
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
4__inference_embedding_ETHNICITY_layer_call_fn_221203�
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
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_221213�
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
2__inference_embedding_MARITAL_layer_call_fn_221220�
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
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_221230�
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
(__inference_flatten_layer_call_fn_221235�
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
C__inference_flatten_layer_call_and_return_conditional_losses_221241�
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
*__inference_flatten_1_layer_call_fn_221246�
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_221252�
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
*__inference_flatten_2_layer_call_fn_221257�
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_221263�
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
*__inference_flatten_3_layer_call_fn_221268�
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_221274�
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
,__inference_concatenate_layer_call_fn_221283�
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
G__inference_concatenate_layer_call_and_return_conditional_losses_221293�
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
&__inference_dense_layer_call_fn_221302�
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
A__inference_dense_layer_call_and_return_conditional_losses_221313�
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
(__inference_dense_1_layer_call_fn_221322�
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
C__inference_dense_1_layer_call_and_return_conditional_losses_221333�
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
'__inference_z_mean_layer_call_fn_221342�
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
B__inference_z_mean_layer_call_and_return_conditional_losses_221352�
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
*__inference_z_log_var_layer_call_fn_221361�
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
E__inference_z_log_var_layer_call_and_return_conditional_losses_221371�
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
"__inference_z_layer_call_fn_221377
"__inference_z_layer_call_fn_221383�
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
=__inference_z_layer_call_and_return_conditional_losses_221405
=__inference_z_layer_call_and_return_conditional_losses_221427�
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
(__inference_encoder_layer_call_fn_220479input_ETHNICITYinput_GENDERinput_MARITAL
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
(__inference_encoder_layer_call_fn_220951inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
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
(__inference_encoder_layer_call_fn_220988inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
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
(__inference_encoder_layer_call_fn_220779input_ETHNICITYinput_GENDERinput_MARITAL
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
C__inference_encoder_layer_call_and_return_conditional_losses_221075inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
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
C__inference_encoder_layer_call_and_return_conditional_losses_221162inputs/input_ETHNICITYinputs/input_GENDERinputs/input_MARITALinputs/input_RACEinputs/input_continuous"�
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
C__inference_encoder_layer_call_and_return_conditional_losses_220827input_ETHNICITYinput_GENDERinput_MARITAL
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
C__inference_encoder_layer_call_and_return_conditional_losses_220875input_ETHNICITYinput_GENDERinput_MARITAL
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
$__inference_signature_wrapper_220914input_ETHNICITYinput_GENDERinput_MARITAL
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
1__inference_embedding_GENDER_layer_call_fn_221169inputs"�
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
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_221179inputs"�
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
/__inference_embedding_RACE_layer_call_fn_221186inputs"�
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
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_221196inputs"�
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
4__inference_embedding_ETHNICITY_layer_call_fn_221203inputs"�
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
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_221213inputs"�
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
2__inference_embedding_MARITAL_layer_call_fn_221220inputs"�
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
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_221230inputs"�
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
(__inference_flatten_layer_call_fn_221235inputs"�
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
C__inference_flatten_layer_call_and_return_conditional_losses_221241inputs"�
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
*__inference_flatten_1_layer_call_fn_221246inputs"�
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_221252inputs"�
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
*__inference_flatten_2_layer_call_fn_221257inputs"�
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_221263inputs"�
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
*__inference_flatten_3_layer_call_fn_221268inputs"�
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
E__inference_flatten_3_layer_call_and_return_conditional_losses_221274inputs"�
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
,__inference_concatenate_layer_call_fn_221283inputs/0inputs/1inputs/2inputs/3inputs/4"�
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
G__inference_concatenate_layer_call_and_return_conditional_losses_221293inputs/0inputs/1inputs/2inputs/3inputs/4"�
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
&__inference_dense_layer_call_fn_221302inputs"�
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
A__inference_dense_layer_call_and_return_conditional_losses_221313inputs"�
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
(__inference_dense_1_layer_call_fn_221322inputs"�
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
C__inference_dense_1_layer_call_and_return_conditional_losses_221333inputs"�
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
'__inference_z_mean_layer_call_fn_221342inputs"�
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
B__inference_z_mean_layer_call_and_return_conditional_losses_221352inputs"�
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
*__inference_z_log_var_layer_call_fn_221361inputs"�
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
E__inference_z_log_var_layer_call_and_return_conditional_losses_221371inputs"�
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
"__inference_z_layer_call_fn_221377inputs/0inputs/1"�
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
"__inference_z_layer_call_fn_221383inputs/0inputs/1"�
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
=__inference_z_layer_call_and_return_conditional_losses_221405inputs/0inputs/1"�
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
=__inference_z_layer_call_and_return_conditional_losses_221427inputs/0inputs/1"�
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
!__inference__wrapped_model_220240�70)"\]delmtu���
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
G__inference_concatenate_layer_call_and_return_conditional_losses_221293����
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
,__inference_concatenate_layer_call_fn_221283����
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
C__inference_dense_1_layer_call_and_return_conditional_losses_221333^de0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense_1_layer_call_fn_221322Qde0�-
&�#
!�
inputs����������
� "������������
A__inference_dense_layer_call_and_return_conditional_losses_221313]\]/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� z
&__inference_dense_layer_call_fn_221302P\]/�,
%�"
 �
inputs���������
� "������������
O__inference_embedding_ETHNICITY_layer_call_and_return_conditional_losses_221213_0/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
4__inference_embedding_ETHNICITY_layer_call_fn_221203R0/�,
%�"
 �
inputs���������
� "�����������
L__inference_embedding_GENDER_layer_call_and_return_conditional_losses_221179_"/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
1__inference_embedding_GENDER_layer_call_fn_221169R"/�,
%�"
 �
inputs���������
� "�����������
M__inference_embedding_MARITAL_layer_call_and_return_conditional_losses_221230_7/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
2__inference_embedding_MARITAL_layer_call_fn_221220R7/�,
%�"
 �
inputs���������
� "�����������
J__inference_embedding_RACE_layer_call_and_return_conditional_losses_221196_)/�,
%�"
 �
inputs���������
� ")�&
�
0���������
� �
/__inference_embedding_RACE_layer_call_fn_221186R)/�,
%�"
 �
inputs���������
� "�����������
C__inference_encoder_layer_call_and_return_conditional_losses_220827�70)"\]delmtu���
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
C__inference_encoder_layer_call_and_return_conditional_losses_220875�70)"\]delmtu���
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
C__inference_encoder_layer_call_and_return_conditional_losses_221075�70)"\]delmtu���
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
C__inference_encoder_layer_call_and_return_conditional_losses_221162�70)"\]delmtu���
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
(__inference_encoder_layer_call_fn_220479�70)"\]delmtu���
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
(__inference_encoder_layer_call_fn_220779�70)"\]delmtu���
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
2����������
(__inference_encoder_layer_call_fn_220951�70)"\]delmtu���
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
2����������
(__inference_encoder_layer_call_fn_220988�70)"\]delmtu���
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_221252\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� }
*__inference_flatten_1_layer_call_fn_221246O3�0
)�&
$�!
inputs���������
� "�����������
E__inference_flatten_2_layer_call_and_return_conditional_losses_221263\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� }
*__inference_flatten_2_layer_call_fn_221257O3�0
)�&
$�!
inputs���������
� "�����������
E__inference_flatten_3_layer_call_and_return_conditional_losses_221274\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� }
*__inference_flatten_3_layer_call_fn_221268O3�0
)�&
$�!
inputs���������
� "�����������
C__inference_flatten_layer_call_and_return_conditional_losses_221241\3�0
)�&
$�!
inputs���������
� "%�"
�
0���������
� {
(__inference_flatten_layer_call_fn_221235O3�0
)�&
$�!
inputs���������
� "�����������
$__inference_signature_wrapper_220914�70)"\]delmtu���
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
=__inference_z_layer_call_and_return_conditional_losses_221405�b�_
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
=__inference_z_layer_call_and_return_conditional_losses_221427�b�_
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
"__inference_z_layer_call_fn_221377~b�_
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
"__inference_z_layer_call_fn_221383~b�_
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
E__inference_z_log_var_layer_call_and_return_conditional_losses_221371]tu0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_z_log_var_layer_call_fn_221361Ptu0�-
&�#
!�
inputs����������
� "�����������
B__inference_z_mean_layer_call_and_return_conditional_losses_221352]lm0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_z_mean_layer_call_fn_221342Plm0�-
&�#
!�
inputs����������
� "����������