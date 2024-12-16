const Layout = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4 lg:max-w-lg xl:max-w-2xl md:max-w-md sm:max-w-sm m-auto">
      {children}
    </div>
  );
};

export default Layout;
