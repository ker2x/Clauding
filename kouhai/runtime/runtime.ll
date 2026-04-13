; --- Kouhai runtime helpers ---

declare i32 @printf(ptr, ...)
declare i32 @puts(ptr)
declare ptr @malloc(i64)
declare i64 @strlen(ptr)
declare ptr @memcpy(ptr, ptr, i64)
declare i32 @snprintf(ptr, i64, ptr, ...)
declare ptr @realloc(ptr, i64)
declare i32 @memcmp(ptr, ptr, i64)
declare ptr @strstr(ptr, ptr)
declare double @llvm.sqrt.f64(double)
declare double @llvm.sin.f64(double)
declare double @llvm.cos.f64(double)
declare double @llvm.tan.f64(double)
declare double @llvm.exp.f64(double)
declare double @llvm.log.f64(double)
declare double @llvm.log2.f64(double)
declare double @llvm.log10.f64(double)
declare double @llvm.pow.f64(double, double)
declare double @llvm.floor.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.round.f64(double)
declare double @llvm.trunc.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.fabs.f64(double)
declare i64 @llvm.abs.i64(i64, i1)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i64 @llvm.ctpop.i64(i64)
declare i64 @llvm.bswap.i64(i64)
declare void @llvm.trap()

%struct.Array = type { i64, i64, ptr }

@_rt_fmt_lld = private unnamed_addr constant [5 x i8] c"%lld\00"
@_rt_fmt_lldn = private unnamed_addr constant [6 x i8] c"%lld\0A\00"
@_rt_fmt_llu = private unnamed_addr constant [5 x i8] c"%llu\00"
@_rt_fmt_llun = private unnamed_addr constant [6 x i8] c"%llu\0A\00"
@_rt_fmt_g = private unnamed_addr constant [3 x i8] c"%g\00"
@_rt_fmt_gn = private unnamed_addr constant [4 x i8] c"%g\0A\00"
@_rt_fmt_pn = private unnamed_addr constant [4 x i8] c"%p\0A\00"
@_rt_str_true = private unnamed_addr constant [5 x i8] c"true\00"
@_rt_str_false = private unnamed_addr constant [6 x i8] c"false\00"

define ptr @_rt_str_concat(ptr %a, ptr %b) {
  %la = call i64 @strlen(ptr %a)
  %lb = call i64 @strlen(ptr %b)
  %total = add i64 %la, %lb
  %total1 = add i64 %total, 1
  %buf = call ptr @malloc(i64 %total1)
  call ptr @memcpy(ptr %buf, ptr %a, i64 %la)
  %dst = getelementptr i8, ptr %buf, i64 %la
  %lb1 = add i64 %lb, 1
  call ptr @memcpy(ptr %dst, ptr %b, i64 %lb1)
  ret ptr %buf
}

define i1 @_rt_str_eq(ptr %a, ptr %b) {
  %la = call i64 @strlen(ptr %a)
  %lb = call i64 @strlen(ptr %b)
  %len_eq = icmp eq i64 %la, %lb
  br i1 %len_eq, label %compare, label %false
compare:
  %cmp = call i32 @memcmp(ptr %a, ptr %b, i64 %la)
  %eq = icmp eq i32 %cmp, 0
  ret i1 %eq
false:
  ret i1 false
}

define void @_rt_print_i64(i64 %v) {
  %fmt = getelementptr [6 x i8], ptr @_rt_fmt_lldn, i64 0, i64 0
  call i32 (ptr, ...) @printf(ptr %fmt, i64 %v)
  ret void
}

define void @_rt_print_u64(i64 %v) {
  %fmt = getelementptr [6 x i8], ptr @_rt_fmt_llun, i64 0, i64 0
  call i32 (ptr, ...) @printf(ptr %fmt, i64 %v)
  ret void
}

define void @_rt_print_double(double %v) {
  %fmt = getelementptr [4 x i8], ptr @_rt_fmt_gn, i64 0, i64 0
  call i32 (ptr, ...) @printf(ptr %fmt, double %v)
  ret void
}

define void @_rt_print_bool(i1 %v) {
  %t = getelementptr [5 x i8], ptr @_rt_str_true, i64 0, i64 0
  %f = getelementptr [6 x i8], ptr @_rt_str_false, i64 0, i64 0
  %s = select i1 %v, ptr %t, ptr %f
  call i32 @puts(ptr %s)
  ret void
}

define void @_rt_print_str(ptr %v) {
  call i32 @puts(ptr %v)
  ret void
}

define void @_rt_print_ptr(ptr %v) {
  %fmt = getelementptr [4 x i8], ptr @_rt_fmt_pn, i64 0, i64 0
  call i32 (ptr, ...) @printf(ptr %fmt, ptr %v)
  ret void
}

define ptr @_rt_i64_to_str(i64 %v) {
  %buf = call ptr @malloc(i64 64)
  %fmt = getelementptr [5 x i8], ptr @_rt_fmt_lld, i64 0, i64 0
  call i32 (ptr, i64, ptr, ...) @snprintf(ptr %buf, i64 64, ptr %fmt, i64 %v)
  ret ptr %buf
}

define ptr @_rt_u64_to_str(i64 %v) {
  %buf = call ptr @malloc(i64 64)
  %fmt = getelementptr [5 x i8], ptr @_rt_fmt_llu, i64 0, i64 0
  call i32 (ptr, i64, ptr, ...) @snprintf(ptr %buf, i64 64, ptr %fmt, i64 %v)
  ret ptr %buf
}

define ptr @_rt_double_to_str(double %v) {
  %buf = call ptr @malloc(i64 64)
  %fmt = getelementptr [3 x i8], ptr @_rt_fmt_g, i64 0, i64 0
  call i32 (ptr, i64, ptr, ...) @snprintf(ptr %buf, i64 64, ptr %fmt, double %v)
  ret ptr %buf
}

define ptr @_rt_bool_to_str(i1 %v) {
  %t = getelementptr [5 x i8], ptr @_rt_str_true, i64 0, i64 0
  %f = getelementptr [6 x i8], ptr @_rt_str_false, i64 0, i64 0
  %s = select i1 %v, ptr %t, ptr %f
  ret ptr %s
}

define ptr @_rt_array_new(i64 %elem_sz) {
  %arr = call ptr @malloc(i64 24)
  %len_ptr = getelementptr %struct.Array, ptr %arr, i32 0, i32 0
  store i64 0, ptr %len_ptr
  %cap_ptr = getelementptr %struct.Array, ptr %arr, i32 0, i32 1
  store i64 8, ptr %cap_ptr
  %buf_sz = mul i64 %elem_sz, 8
  %data = call ptr @malloc(i64 %buf_sz)
  %data_ptr = getelementptr %struct.Array, ptr %arr, i32 0, i32 2
  store ptr %data, ptr %data_ptr
  ret ptr %arr
}

define void @_rt_array_ensure_cap(ptr %arr, i64 %elem_sz) {
  %len_ptr = getelementptr %struct.Array, ptr %arr, i32 0, i32 0
  %len = load i64, ptr %len_ptr
  %cap_ptr = getelementptr %struct.Array, ptr %arr, i32 0, i32 1
  %cap = load i64, ptr %cap_ptr
  %need = icmp sge i64 %len, %cap
  br i1 %need, label %grow, label %done
grow:
  %new_cap = mul i64 %cap, 2
  store i64 %new_cap, ptr %cap_ptr
  %new_bytes = mul i64 %new_cap, %elem_sz
  %data_ptr = getelementptr %struct.Array, ptr %arr, i32 0, i32 2
  %old_data = load ptr, ptr %data_ptr
  %new_data = call ptr @realloc(ptr %old_data, i64 %new_bytes)
  store ptr %new_data, ptr %data_ptr
  br label %done
done:
  ret void
}

define i64 @kouhai_str_len(ptr %s) {
  %len = call i64 @strlen(ptr %s)
  ret i64 %len
}

define i64 @kouhai_str_char_at(ptr %s, i64 %i) {
  %ptr = getelementptr i8, ptr %s, i64 %i
  %ch = load i8, ptr %ptr
  %val = zext i8 %ch to i64
  ret i64 %val
}

define ptr @kouhai_str_substring(ptr %s, i64 %start, i64 %end) {
  %len = sub i64 %end, %start
  %buf_sz = add i64 %len, 1
  %buf = call ptr @malloc(i64 %buf_sz)
  %src = getelementptr i8, ptr %s, i64 %start
  call ptr @memcpy(ptr %buf, ptr %src, i64 %len)
  %null_ptr = getelementptr i8, ptr %buf, i64 %len
  store i8 0, ptr %null_ptr
  ret ptr %buf
}

define i1 @kouhai_str_starts_with(ptr %s, ptr %prefix) {
  %slen = call i64 @strlen(ptr %s)
  %plen = call i64 @strlen(ptr %prefix)
  %too_short = icmp slt i64 %slen, %plen
  br i1 %too_short, label %no, label %check
check:
  %cmp = call i32 @memcmp(ptr %s, ptr %prefix, i64 %plen)
  %eq = icmp eq i32 %cmp, 0
  ret i1 %eq
no:
  ret i1 false
}

define i64 @kouhai_str_index_of(ptr %s, ptr %needle) {
  %found = call ptr @strstr(ptr %s, ptr %needle)
  %is_null = icmp eq ptr %found, null
  br i1 %is_null, label %notfound, label %calc
calc:
  %s_int = ptrtoint ptr %s to i64
  %f_int = ptrtoint ptr %found to i64
  %idx = sub i64 %f_int, %s_int
  ret i64 %idx
notfound:
  ret i64 -1
}

define ptr @kouhai_str_from_char(i64 %code) {
  %buf = call ptr @malloc(i64 2)
  %ch = trunc i64 %code to i8
  store i8 %ch, ptr %buf
  %null_ptr = getelementptr i8, ptr %buf, i64 1
  store i8 0, ptr %null_ptr
  ret ptr %buf
}

define double @kouhai_sqrt(double %v) {
  %r = call double @llvm.sqrt.f64(double %v)
  ret double %r
}

define double @kouhai_sin(double %v) {
  %r = call double @llvm.sin.f64(double %v)
  ret double %r
}

define double @kouhai_cos(double %v) {
  %r = call double @llvm.cos.f64(double %v)
  ret double %r
}

define double @kouhai_tan(double %v) {
  %r = call double @llvm.tan.f64(double %v)
  ret double %r
}

define double @kouhai_exp(double %v) {
  %r = call double @llvm.exp.f64(double %v)
  ret double %r
}

define double @kouhai_log(double %v) {
  %r = call double @llvm.log.f64(double %v)
  ret double %r
}

define double @kouhai_log2(double %v) {
  %r = call double @llvm.log2.f64(double %v)
  ret double %r
}

define double @kouhai_log10(double %v) {
  %r = call double @llvm.log10.f64(double %v)
  ret double %r
}

define double @kouhai_floor(double %v) {
  %r = call double @llvm.floor.f64(double %v)
  ret double %r
}

define double @kouhai_ceil(double %v) {
  %r = call double @llvm.ceil.f64(double %v)
  ret double %r
}

define double @kouhai_round(double %v) {
  %r = call double @llvm.round.f64(double %v)
  ret double %r
}

define double @kouhai_trunc(double %v) {
  %r = call double @llvm.trunc.f64(double %v)
  ret double %r
}

define double @kouhai_abs(double %v) {
  %r = call double @llvm.fabs.f64(double %v)
  ret double %r
}

define double @kouhai_pow(double %a, double %b) {
  %r = call double @llvm.pow.f64(double %a, double %b)
  ret double %r
}

define double @kouhai_fma(double %a, double %b, double %c) {
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %r
}

define i64 @kouhai_abs_i64(i64 %v) {
  %r = call i64 @llvm.abs.i64(i64 %v, i1 true)
  ret i64 %r
}

define i64 @kouhai_clz(i64 %v) {
  %r = call i64 @llvm.ctlz.i64(i64 %v, i1 false)
  ret i64 %r
}

define i64 @kouhai_ctz(i64 %v) {
  %r = call i64 @llvm.cttz.i64(i64 %v, i1 false)
  ret i64 %r
}

define i64 @kouhai_popcount(i64 %v) {
  %r = call i64 @llvm.ctpop.i64(i64 %v)
  ret i64 %r
}

define i64 @kouhai_bswap(i64 %v) {
  %r = call i64 @llvm.bswap.i64(i64 %v)
  ret i64 %r
}

define void @kouhai_panic() {
  call void @llvm.trap()
  unreachable
}

