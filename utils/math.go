package utils

func MultiplyAll[v int | int8 | int16 | int32 | int64 | float32 | float64](s []v) v {
	var prod v = 1
	for _, value := range s {
		prod *= value
	}
	return prod
}
