package main

func main() {
	fapi := FaceAPI{}
	err := fapi.init()
	if err != nil {
		panic(err)
	}
}
