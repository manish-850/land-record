import { Link } from "react-router-dom";

const NavLink = ({ href, text }) => {
  return (
    <Link
      to={href}
      className="text-sm font-medium text-neutral-800 transition-colors duration-200 hover:text-black"
    >
      {text}
    </Link>
  );
};

export default NavLink;
