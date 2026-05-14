import cls from './Header.module.scss';
import LogoIcon from '../../shared/assets/icons/CTReport.svg';
import UserIcon from '../../shared/assets/icons/userIcon.svg';

export const Header = () => {
  return (
    <header className={cls.header}>
      <div className={cls.left}>
        <img className={cls.logoIcon} src={LogoIcon} alt="HomeIcon" />
      </div>

      <div className={cls.right}>
      <img className={cls.userIcon} src={UserIcon} alt="HomeIcon" />
      </div>
    </header>
  );
};