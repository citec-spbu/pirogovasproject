import { useNavigate } from 'react-router-dom';
import { ProfileDropdown } from '../ProfileDropdown/ProfileDropdown';
import { logoutUser } from '../../shared/api/authApi';
import HeartlogoIcon from '../../shared/assets/icons/heartLogoIcon.svg';
import cls from './Header.module.scss';

export const Header = () => {
  const navigate = useNavigate();

  const handleLogout = () => {
    logoutUser();
    navigate('/login', { replace: true });
  };

  return (
    <header className={cls.header}>
      <div className={cls.logoBlock}>
          <img src={HeartlogoIcon} alt="CTReportUser" className={cls.logoIcon} />
          <span className={cls.logoText}>CTReport</span>
        </div>

      <ProfileDropdown
        triggerLabel="Пользователь"
        fullName="Фамилия Имя Отчество"
        roleLabel="Пользователь"
        onLogout={handleLogout}
      />
    </header>
  );
};