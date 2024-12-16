import { Outlet } from 'react-router-dom';
import Layout from './Layout';

const RootLayout = () => {
  return (
    <Layout>
      <Outlet />
    </Layout>
  );
};

export default RootLayout;
