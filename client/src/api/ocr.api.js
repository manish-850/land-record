import axios from 'axios';

const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

const api = axios.create({
    baseURL: `${backendUrl}/api/ocr`,
    withCredentials: true
})



// register
export const extractionHandler = async (data) => {
    const response = await api.post("/extract", data);
    return response.data;
}