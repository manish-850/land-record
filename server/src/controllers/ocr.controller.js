import axios from "axios";
const api = axios.create({
  baseURL: process.env.ML_SERVER_URL || "http://localhost:8000",
  withCredentials: true,
});

export const extractDataController = async (req, res) => {
  console.log(req.file);
  const formData = new FormData();

  const blob = new Blob([req.file.buffer], {
    type: req.file.mimetype,
  });

  formData.append("file", blob, req.file.originalname);
  try {
    const { data } = await api.post("/extract", formData);
    res.status(200).json({
      message: "extracted successfully",
      data,
    });
  } catch (err) {
    console.log(err);
    res.status(500).json({
      message: "extraction failed",
      err,
    });
  }
};

export default extractDataController;
