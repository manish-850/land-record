const Record = ({field, value}) => {
  return (
    <div className="w-[80%] h-10">
      <div className="flex w-full justify-between">
        <p className="w-1/3 text-sm">{field}</p>
        <p className="w-1/3 text-sm">{value.value}</p>
        <div className="flex gap-8 w-1/3 justify-between bg-purple-500">
          <p className="flex-1">{value.confidence}</p>
          <p className="flex-1">{value.confidence_status}</p>
        </div>
      </div>
    </div>
  );
};

export default Record;
