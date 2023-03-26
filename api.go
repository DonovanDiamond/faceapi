package main

import (
	"encoding/base64"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

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
			Data     string `json:"data"`
			Multiple bool   `json:"multiple"`
		}

		if err := c.BodyParser(&d); err != nil {
			return errors.Wrap(err, "failed to parse body")
		}

		raw, err := base64.StdEncoding.DecodeString(d.Data)
		if err != nil {
			return errors.Wrap(err, "failed to decode base64")
		}

		var results []int
		if d.Multiple {
			results, err = a.recongizeMultiple(raw)
		} else {
			results = append(results, 0)
			results[0], err = a.recognize(raw)
		}

		return c.JSON(fiber.Map{
			"results": results,
			"error":   err,
		})
	})

	app.Post("/add", func(c *fiber.Ctx) error {
		var d struct {
			Data string `json:"data"`
			ID   int    `json:"id"`
		}

		if err := c.BodyParser(&d); err != nil {
			return errors.Wrap(err, "failed to parse body")
		}

		raw, err := base64.StdEncoding.DecodeString(d.Data)
		if err != nil {
			return errors.Wrap(err, "failed to decode base64")
		}

		filename := fmt.Sprintf("%d-%d.jpg", d.ID, time.Now().Unix())

		err = os.WriteFile(filepath.Join(trainingDir, filename), raw, fs.ModeAppend)
		if err != nil {
			return errors.Wrap(err, "failed to write output file")
		}

		return c.JSON(fiber.Map{
			"filename": filename,
			"id":       d.ID,
		})
	})

	app.Post("/train", func(c *fiber.Ctx) error {
		err := a.train()

		return c.JSON(fiber.Map{
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
		face, err := a.rec.RecognizeSingleFile(filepath.Join(trainingDir, file.Name()))
		if err != nil {
			return errors.Wrapf(err, "failed to recognize file %s", file.Name())
		}
		if face == nil {
			log.Printf("could not find face on %s", file.Name())
			continue
		}
		id, err := strconv.Atoi(strings.Split(file.Name(), "-")[0])
		if err != nil {
			return errors.Wrap(err, "failed to parse file name")
		}
		cats = append(cats, int32(id))
		samples = append(samples, face.Descriptor)
		log.Printf("\t training %s...", file.Name())
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

func (a *FaceAPI) recongizeMultiple(data []byte) (results []int, err error) {
	faces, err := a.rec.Recognize(data)
	if err != nil {
		err = errors.Wrap(err, "failed to recongize face")
	}

	for _, face := range faces {
		results = append(results, a.rec.Classify(face.Descriptor))
	}

	return
}
