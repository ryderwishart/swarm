import { Outlet } from 'react-router-dom';

const RootLayout = () => {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900/50">
      <main className="mx-auto max-w-[90rem] h-full min-h-screen flex flex-col">
        <Outlet />
      </main>
    </div>
  );
};

export default RootLayout;
