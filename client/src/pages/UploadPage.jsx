import { InputField } from "@/components/auth/InputField";
import { Button } from "@/components/ui/button";
import { Upload, FileText, Trash, View, X } from "lucide-react";
import { useRef, useState } from "react";

const UploadPage = () => {
  const fileRef = useRef(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [file, setFile] = useState(null);
  const [isPreviewed, setIsPreviewed] = useState(false);
  return (
    <div className="flex h-screen w-full items-center justify-center">
      {file && isPreviewed && (
        <div className="w-full h-full absolute top-0 left-0 z-99 flex justify-center items-center py-5">
          <iframe
            target="_blanck"
            src={`${fileUrl}#toolbar=0&navpanes=0`}
            title="PDF Preview"
            className="w-1/2 h-full border-none flex justify-center"
          />
          <Button
            variant="destructive"
            size="icon"
            className="absolute top-8 right-8"
            onClick={() => setIsPreviewed(false)}
          >
            <X strokeWidth={2} />
          </Button>
        </div>
      )}
      <div className="w-100 h-100 shadow-xl bg-[#f8ffe8] rounded-xl border overflow-hidden relative">
        <div className="flex flex-col w-full h-full gap-6 px-8 py-4">
          <div>
            <h4>Add new land record</h4>
            <p className="text-sm opacity-80">
              Drag and drop files to add new land record.
            </p>
          </div>

          <div className="w-full flex-1 flex flex-col gap-2">
            <div
              onClick={() => fileRef.current.click()}
              className="flex-1 flex flex-col items-center justify-center gap-4 border border-dashed rounded cursor-pointer"
            >
              <InputField
                onChange={(e) => {
                  setFile(e.target.files[0]);
                  setFileUrl(URL.createObjectURL(e.target.files[0]));
                }}
                ref={fileRef}
                type="file"
                className="hidden"
              />
              <Upload strokeWidth={1} size={65} />
              <p className="opacity-80">Drag and drop or choose your file</p>
            </div>
            {file && (
              <>
                <div className="w-full h-15 bg-zinc-100 rounded flex justify-between items-center px-2">
                  <div className="flex gap-2 items-center h-full">
                    <div className="flex items-center justify-center h-full">
                      <FileText strokeWidth={1} size={50} />
                    </div>
                    <div>
                      <p>{file.name}</p>
                      <p className="text-xs opacity-80">
                        {Math.round(file.size / 1024)} KB
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={() => setFile(null)}
                      variant="destructive"
                      size="icon"
                    >
                      <Trash strokeWidth={1} />
                    </Button>
                    <Button onClick={() => setIsPreviewed(true)} size="icon">
                      <View strokeWidth={1} />
                    </Button>
                  </div>
                </div>
              </>
            )}

            <div className="w-full flex items-center justify-end gap-4">
              <Button variant="secondary" className="bg-[#d2e99c]">
                Cancel
              </Button>
              <Button className="bg-[#84994F]">Upload</Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;
