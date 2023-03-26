package main

import (
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	face "github.com/Kagami/go-face"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/pkg/errors"
)

const dataDir = "data"

var (
	modelsDir   = filepath.Join(dataDir, "models")
	trainingDir = filepath.Join(dataDir, "training")
)

type FaceAPI struct {
	rec *face.Recognizer
}

func (a *FaceAPI) init() (err error) {
	log.Println("Creating recognizer...")
	a.rec, err = face.NewRecognizer(modelsDir)
	if err != nil {
		return errors.Wrap(err, "failed to create new recognizer")
	}

	log.Println("Creating router...")
	app := fiber.New()

	app.Use(logger.New())

	app.Post("/recognize", func(c *fiber.Ctx) error {
		var d struct {
			Data []byte `json:"data"`
		}

		if err := c.BodyParser(&d); err != nil {
			return errors.Wrap(err, "failed to parse body")
		}

		id, err := a.recognize(d.Data)
		return c.JSON(fiber.Map{
			"id":    id,
			"error": err,
		})
	})

	log.Println("Training...")
	err = a.train()
	if err != nil {
		return errors.Wrap(err, "failed to train")
	}

	log.Println("listening...")
	return app.Listen(":1234")
}

func (a *FaceAPI) train() (err error) {
	files, err := os.ReadDir(trainingDir)
	if err != nil {
		return errors.Wrap(err, "failed to read training dir")
	}
	var samples []face.Descriptor
	var cats []int32
	for _, file := range files {
		log.Printf("\t training %s...", file.Name())
		faces, err := a.rec.RecognizeFile(filepath.Join(trainingDir, file.Name()))
		if err != nil {
			return errors.Wrapf(err, "failed to recognize file %s", file.Name())
		}
		if len(faces) != 1 {
			return errors.Wrapf(err, "there isnt 1 face on %s", file.Name())
		}
		id, _ := strconv.Atoi(strings.Split(file.Name(), "-")[0])
		cats = append(cats, int32(id))
		samples = append(samples, faces[0].Descriptor)
	}
	a.rec.SetSamples(samples, cats)
	return
}

func (a *FaceAPI) recognize(data []byte) (result int, err error) {
	face, err := a.rec.RecognizeSingle(data)
	if err != nil {
		return 0, errors.Wrap(err, "failed to recongize face")
	}

	if face == nil {
		return 0, errors.Wrap(err, "no face on image")
	}
	id := a.rec.Classify(face.Descriptor)
	return id, nil
}
