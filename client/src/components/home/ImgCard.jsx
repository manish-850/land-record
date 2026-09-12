const ImgCard = ({ src, alt, className }) => {
  return (
    <div
      className={`bg-neutral-100 rounded-lg overflow-hidden border border-neutral-200 relative flex items-center justify-center ${className}`}
    >
      <img
        src={src}
        alt={alt}
        className="w-full h-full object-cover block error-hide"
        onError={(e) => {
          e.currentTarget.style.display = "none";
          if (e.currentTarget.parentElement) {
            e.currentTarget.parentElement.innerText = "Card 1";
          }
        }}
      />
    </div>
  );
};

export default ImgCard;
