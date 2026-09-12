import NavLink from "./NavLink";

const Navbar = () => {
  return (
    <nav className="flex w-full items-center justify-between">
      <h1 className="text-md font-bold text-black ">Bhumi Setu</h1>
      <div className="flex items-center gap-10">
        <NavLink href="/" text="Home" />
        <NavLink href="/" text="Contact us" />
        <NavLink href="/" text="About us" />
      </div>
    </nav>
  );
};

export default Navbar;
